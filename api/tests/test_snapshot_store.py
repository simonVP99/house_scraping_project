"""Snapshot store: payload extraction, lossless round-trip, and safe file naming."""
import glob
import gzip
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from snapshot_store import (  # noqa: E402
    SnapshotStore, extract_classified_payload, read_store, static_fingerprint,
    store_path,
)


def _sample_html():
    files = sorted(glob.glob(os.path.join(ROOT, 'data', '*', 'full_source_*.html')))
    if not files:
        pytest.skip('no scraped HTML available')
    with open(files[0], encoding='utf-8', errors='replace') as fh:
        return fh.read()


def test_extracts_payload_from_real_page():
    payload = extract_classified_payload(_sample_html())
    assert isinstance(payload, dict)
    assert 'id' in payload and 'property' in payload


def test_returns_none_for_non_listing_pages():
    assert extract_classified_payload('<html>captcha-delivery</html>') is None
    # Truncated JSON must not raise; it means an interrupted download.
    assert extract_classified_payload('<script>window.classified = {"id": 1') is None


def test_raw_decode_survives_braces_inside_strings():
    # A description containing '};' used to truncate a non-greedy regex.
    page = '<script>window.classified = {"id": 7, "desc": "a }; b { c"};</script>'
    assert extract_classified_payload(page) == {'id': 7, 'desc': 'a }; b { c'}


def test_store_round_trip_is_lossless(tmp_path):
    html = _sample_html()
    store = SnapshotStore(str(tmp_path))
    assert store.save('123456', '01012025', html) == 'stored'
    assert store.save('123456', '02012025', html) == 'stored'
    records = read_store(store_path(str(tmp_path), '123456'))
    assert [r['date'] for r in records] == ['01012025', '02012025']
    # The payload kept in the store is byte-identical to the one in the page.
    assert json.dumps(records[0]['js'], sort_keys=True) == json.dumps(
        extract_classified_payload(html), sort_keys=True)


def test_same_date_replaces_rather_than_duplicates(tmp_path):
    html = _sample_html()
    store = SnapshotStore(str(tmp_path))
    store.save('123456', '01012025', html)
    assert store.save('123456', '01012025', html) == 'replaced'
    assert len(read_store(store_path(str(tmp_path), '123456'))) == 1


def test_pages_without_payload_are_not_stored(tmp_path):
    store = SnapshotStore(str(tmp_path))
    assert store.save('999', '01012025', '<html>blocked</html>') == 'no_payload'
    assert read_store(store_path(str(tmp_path), '999')) == []


def test_static_fingerprint_ignores_volatile_counters():
    base = {'id': 1, 'statistics': {'viewCount': 10, 'bookmarkCount': 2},
            'price': {'mainValue': 300000}}
    later = {'id': 1, 'statistics': {'viewCount': 99, 'bookmarkCount': 40},
             'price': {'mainValue': 300000}}
    assert static_fingerprint(base) == static_fingerprint(later)
    changed = {'id': 1, 'statistics': {'viewCount': 99, 'bookmarkCount': 40},
               'price': {'mainValue': 310000}}
    assert static_fingerprint(base) != static_fingerprint(changed)


def test_store_is_gzipped_on_disk(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.save('123456', '01012025', _sample_html())
    path = store_path(str(tmp_path), '123456')
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        assert json.loads(fh.readline())['date'] == '01012025'


def test_store_parse_matches_html_parse(tmp_path):
    """A row rebuilt from the store must equal the row built from the page."""
    from parser import build_row, extract_classified_json, parse_file_identity

    files = sorted(glob.glob(os.path.join(ROOT, 'data', '*', 'full_source_*.html')))[:25]
    if not files:
        pytest.skip('no scraped HTML available')

    store = SnapshotStore(str(tmp_path))
    for path in files:
        property_id, date_tag = parse_file_identity(path)
        with open(path, encoding='utf-8', errors='replace') as fh:
            html = fh.read()
        store.save(property_id, date_tag, html)
        from_html = build_row(extract_classified_json(html), property_id, date_tag)
        # The 25 sampled files can include several snapshots of one property,
        # so select the record by date rather than assuming it is first.
        record = next(r for r in read_store(store_path(str(tmp_path), property_id))
                      if r['date'] == date_tag)
        from_store = build_row(record['js'], property_id, date_tag)
        assert from_store == from_html


def test_store_parse_round_trips_every_field(tmp_path):
    from parser import parse_snapshot_store

    files = sorted(glob.glob(os.path.join(ROOT, 'data', '*', 'full_source_*.html')))[:40]
    if not files:
        pytest.skip('no scraped HTML available')
    store = SnapshotStore(str(tmp_path))
    for path in files:
        from parser import parse_file_identity
        property_id, date_tag = parse_file_identity(path)
        with open(path, encoding='utf-8', errors='replace') as fh:
            store.save(property_id, date_tag, fh.read())

    frame, errors = parse_snapshot_store(str(tmp_path), progress_every=0,
                                         log=lambda *a: None)
    assert errors == []
    assert frame.height == 40
    assert 'property_id' in frame.columns
