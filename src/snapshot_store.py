"""Storage layer for scraped listing snapshots.

Why this exists
---------------
The corpus is 143,806 HTML files across ~30,300 property directories (52 GB).
Only ~5.3% of each file is the ``window.classified`` payload the parser actually
reads: the rest is page chrome, inline scripts, CSS and repeated boilerplate.
Each file is ~363 KB, of which ~36 KB is the payload.

Measured on 120 multi-snapshot properties (570 consecutive snapshot pairs):

* 0.0% of consecutive snapshots are byte-identical, so plain gzip-and-share
  deduplication saves nothing -- the files really do differ.
* Ignoring the volatile keys (``statistics.viewCount``, ``bookmarkCount``,
  ``lastModificationDate``, ``flags.isUnderOption``, media URLs), 56% of
  consecutive snapshots carry an identical payload. The differences are almost
  entirely counters and a re-rendered page, not listing content.
* gzip reaches 21.4% of raw on a single snapshot, and the classifier payload is
  5.3% of raw.

So the win is not "skip duplicate files" (an earlier assumption of mine that
this measurement refutes). It is that the *payload* is small and highly
redundant across snapshots of one property, while the *page* is large and
mostly noise.

Storage format
--------------
Per property, one file:

    <data>/.snapshots/<property_id>.jsonl.gz

Each line is one snapshot::

    {"date": "01062024", "html_sha256": "...", "js": {...classified JSON...},
     "html_len": 371244}

The payload is gzipped as one stream per property, so gzip's 32 KB window sees
all snapshots of that listing at once and collapses the repeated structure that
a per-file ``.html.gz`` cannot.

This module is deliberately standalone: it has no dependency on the scraper,
the driver, or pandas/polars, so it is safe to import from both the py3.9
scraping venv and the py3.13 API venv.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile

CLASSIFIED_MARKER = 'window.classified'

# Keys that legitimately change between snapshots without the listing changing.
# Used only for reporting how much of the corpus is redundant, never for
# deciding what to store -- every snapshot is kept.
VOLATILE_KEYS = frozenset({
    'viewCount', 'bookmarkCount', 'lastModificationDate', 'isUnderOption',
    'isSoldOrRented', 'isNewPrice', 'alertPrice', 'wasPropertyVisited',
    'epcUrl', 'smallUrl', 'mediumUrl', 'largeUrl', 'page404',
})


# ---------------------------------------------------------------------------
# Payload extraction

def extract_classified_payload(html: str):
    """Return the ``window.classified`` object from a page, or None.

    Scans raw text rather than parsing the DOM: the payload is a JSON literal in
    an inline script, and a full HTML parse costs ~26 ms/file versus ~0.5 ms for
    a string find plus ``raw_decode``.

    ``raw_decode`` is used rather than a regex because a description containing
    ``};`` would truncate a non-greedy match at the wrong brace.
    """
    marker = html.find(CLASSIFIED_MARKER)
    while marker != -1:
        brace = html.find('{', marker)
        if brace == -1:
            return None
        try:
            payload, _ = json.JSONDecoder().raw_decode(html[brace:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            # A later marker occurrence may hold the real payload.
            pass
        marker = html.find(CLASSIFIED_MARKER, marker + len(CLASSIFIED_MARKER))
    return None


def _strip_volatile(obj):
    """Recursively drop volatile keys, for redundancy reporting only."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def static_fingerprint(payload) -> str:
    """sha256 over the payload with volatile keys removed."""
    blob = json.dumps(_strip_volatile(payload), sort_keys=True,
                      separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Per-property stores

def store_path(root: str, property_id: str) -> str:
    return os.path.join(root, '.snapshots', f'{property_id}.jsonl.gz')


def read_store(path: str):
    """Load every snapshot of one property, oldest first."""
    if not os.path.exists(path):
        return []
    records = []
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated line means an interrupted append; keep the rest
                # rather than losing the whole property.
                continue
    records.sort(key=lambda r: r.get('date') or '')
    return records


def write_store(path: str, records) -> int:
    """Atomically rewrite a property store. Returns bytes written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get('date') or '')
    tmp = f'{path}.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=9) as handle:
        for record in ordered:
            handle.write(json.dumps(record, separators=(',', ':')))
            handle.write('\n')
    os.replace(tmp, path)
    return os.path.getsize(path)


class SnapshotStore:
    """Append-only snapshot history for the whole corpus.

    One gzipped file per property. Appending re-writes that property's stream
    (it holds only a handful of snapshots, so this is cheap) which keeps the
    compression window packed and avoids gzip's "multiple members" overhead.
    """

    def __init__(self, root: str):
        self.root = root
        self.dir = os.path.join(root, '.snapshots')
        os.makedirs(self.dir, exist_ok=True)

    def save(self, property_id: str, date: str, html: str):
        """Persist one snapshot. Returns a short status string.

        Status is 'stored', 'replaced' (same date already present) or
        'no_payload' (page was a challenge/redirect and carries nothing usable).
        """
        payload = extract_classified_payload(html)
        if payload is None:
            return 'no_payload'

        path = store_path(self.root, property_id)
        records = read_store(path)
        record = {
            'date': date,
            'html_sha256': hashlib.sha256(html.encode('utf-8')).hexdigest(),
            'html_len': len(html),
            'js': payload,
        }
        status = 'stored'
        for index, existing in enumerate(records):
            if existing.get('date') == date:
                records[index] = record
                status = 'replaced'
                break
        else:
            records.append(record)
        write_store(path, records)
        return status

    def snapshot_dates(self, property_id: str):
        return [r.get('date') for r in read_store(store_path(self.root, property_id))]


# ---------------------------------------------------------------------------
# Backfill

def _iter_html_files(data_dir: str):
    """Yield (property_id, date_tag, path) for every snapshot on disk."""
    with os.scandir(data_dir) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name.startswith('.'):
                continue
            property_id = entry.name
            with os.scandir(entry.path) as files:
                for item in files:
                    name = item.name
                    if not (item.is_file(follow_symlinks=False)
                            and name.startswith('full_source_')
                            and name.endswith('.html')):
                        continue
                    date_tag = name[len('full_source_'):-len('.html')]
                    yield property_id, date_tag, item.path


def backfill(data_dir: str, limit_properties=None, keep_html=True,
             progress_every=2000, log=print):
    """Build per-property snapshot stores from existing HTML files.

    ``keep_html=False`` moves the original .html files into a single tar archive
    after they have been absorbed, then removes the directories. That is the
    space-reclaiming path; it is destructive, so it is opt-in and the archive is
    verified to be readable before anything is deleted.
    """
    store = SnapshotStore(data_dir)
    stats = {'properties': 0, 'snapshots': 0, 'no_payload': 0,
             'html_bytes': 0, 'store_bytes': 0}
    grouped = {}
    for property_id, date_tag, path in _iter_html_files(data_dir):
        grouped.setdefault(property_id, []).append((date_tag, path))
        if limit_properties and len(grouped) >= limit_properties:
            break

    for property_id, items in grouped.items():
        records = []
        for date_tag, path in sorted(items):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    html = fh.read()
            except OSError:
                continue
            stats['html_bytes'] += len(html.encode('utf-8'))
            payload = extract_classified_payload(html)
            if payload is None:
                stats['no_payload'] += 1
                continue
            records.append({
                'date': date_tag,
                'html_sha256': hashlib.sha256(html.encode('utf-8')).hexdigest(),
                'html_len': len(html),
                'js': payload,
            })
        if records:
            stats['store_bytes'] += write_store(
                store_path(data_dir, property_id), records)
            stats['snapshots'] += len(records)
        stats['properties'] += 1
        if stats['properties'] % progress_every == 0:
            log(f"  {stats['properties']:,} properties, "
                f"{stats['snapshots']:,} snapshots, "
                f"{(stats['html_bytes'] - stats['store_bytes'])/1e9:.1f} GB saved so far")

    if not keep_html:
        log('  archiving original HTML before removal...')
        archive = os.path.join(data_dir, '..', 'outputs', 'data_html_archive.tar')
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        with tarfile.open(archive, 'w') as tar:
            for property_id, items in grouped.items():
                for date_tag, path in items:
                    tar.add(path, arcname=os.path.join(property_id,
                                                       os.path.basename(path)))
        # Verify the archive lists every file before deleting anything.
        with tarfile.open(archive, 'r') as tar:
            archived = sum(1 for _ in tar)
        expected = sum(len(v) for v in grouped.values())
        if archived != expected:
            raise RuntimeError(
                f'archive verification failed: {archived} entries in the tar, '
                f'{expected} files on disk -- refusing to delete')
        log(f'  archive verified ({archived:,} files), removing HTML tree')
        for property_id, items in grouped.items():
            shutil.rmtree(os.path.join(data_dir, property_id), ignore_errors=True)

    stats['saved_bytes'] = stats['html_bytes'] - stats['store_bytes']
    return stats


def redundancy_report(data_dir: str, limit_properties=300):
    """How much of the corpus is re-observation, for planning decisions."""
    store = SnapshotStore(data_dir)
    pairs = 0
    identical_static = 0
    properties = 0
    for property_id, _date, _path in _iter_html_files(data_dir):
        entries = read_store(store_path(data_dir, property_id))
        properties += 1
        previous = None
        for entry in entries:
            fingerprint = static_fingerprint(entry['js'])
            if previous is not None:
                pairs += 1
                if fingerprint == previous:
                    identical_static += 1
            previous = fingerprint
        if limit_properties and properties >= limit_properties:
            break
    return {
        'properties': properties,
        'consecutive_pairs': pairs,
        'identical_ignoring_volatile': identical_static,
        'redundant_pct': (identical_static / pairs * 100) if pairs else 0.0,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build per-property gzipped snapshot stores from scraped HTML.')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--limit-properties', type=int, default=None,
                        help='only convert the first N properties (for a trial run)')
    parser.add_argument('--remove-html', action='store_true',
                        help='archive then delete the original .html files once stored')
    parser.add_argument('--report-only', action='store_true',
                        help='measure redundancy without writing anything')
    args = parser.parse_args()

    if args.report_only:
        print(json.dumps(redundancy_report(args.data_dir,
                                           args.limit_properties or 300), indent=2))
    else:
        result = backfill(args.data_dir,
                          limit_properties=args.limit_properties,
                          keep_html=not args.remove_html)
        print(json.dumps(result, indent=2))
