# Runbook: Kicking Off a Scrape

How to run a scrape of immoweb.be through this project, and what to expect.

## TL;DR

```bash
cd <project root>
.venv/bin/python -u src/main.py <num_pages>
```

Long runs belong in the background — see [Running it unattended](#running-it-unattended).

---

## Which environment

There are **two** virtualenvs. Using the wrong one is the most common mistake.

| Env | Python | Use for |
|---|---|---|
| `.venv` | 3.9 | **Scraping + parsing** (`src/main.py`, `scraper.py`, `parser.py`) |
| `.venv-api` | 3.13 | Flask API, AutoGluon/torch/SHAP inference (`api/`) |

`.venv` has `undetected_chromedriver` + Selenium; `.venv-api` does not.

```bash
.venv/bin/python -u src/main.py 20      # correct
.venv-api/bin/python -u src/main.py 20  # ModuleNotFoundError: undetected_chromedriver
```

## What one run does

`src/main.py <num_pages>` runs two stages in sequence:

1. **Scrape** — `scraper.scrape_houses(num_pages)` drives a real Chrome via
   `undetected_chromedriver`, saving each listing's HTML to
   `data/<property_id>/full_source_<DDMMYYYY>.html`.
2. **Parse** — `parser.parse_data_folder_parallel()` walks **all** of `data/`
   (not just today's files) and writes `outputs/property_data_raw.csv`.

Note that stage 2 re-parses the entire 51 GB corpus every time, which is
wasteful for an incremental scrape. If you only want today's new listings,
scrape first and then use `src/parse_today_and_append.py` (see
[Parsing only today's files](#parsing-only-todays-files)).

## Politeness / rate limiting

The scraper is deliberately slow. Current delays in `src/scraper.py`:

| Point | Delay |
|---|---|
| Per listing page load | `uniform(4, 7)` s |
| Between listings | `uniform(5, 10)` s |
| Every 10th listing | `uniform(15, 25)` s instead |
| Search results page load | `uniform(8, 12)` s |
| Between search pages | `uniform(10, 20)` s |
| After a DataDome block | `uniform(60, 90)` s backoff |
| After an error | `uniform(30, 60)` s backoff |

That is roughly **13 s per listing**. With ~91 listings per search page that
is **~20 minutes per page**.

Block handling: if a page contains `captcha-delivery` or is under 2000 bytes,
it is treated as blocked. The HTML is dumped to `debug_blocked_page.html`,
the scraper backs off, and it **stops after 3 consecutive blocks** to avoid an
IP ban. Three consecutive errors also stop it.

### Sizing a run

| Pages | Listings | Wall clock | Disk |
|---|---|---|---|
| 1 | ~91 | ~20 min | ~46 MB |
| 10 | ~910 | ~3.5 h | ~460 MB |
| 20 | ~1,800 | ~7 h | ~900 MB |
| 100 | ~9,100 | ~35 h | ~4.6 GB |

Measured average: **504 KB per listing**. `data/` is already 51 GB of history.
Check free space before a big run:

```bash
df -h .
```

### Things this scraper cannot do

- **Run headless.** DataDome detects headless Chrome; a visible browser window is
  used on purpose. The window stays open for the whole run and closes when the
  process exits — this is expected, not a crash.
- **Run without Chrome installed.** It reads the installed major version from
  `/Applications/Google Chrome.app` so the driver matches.
- **Resume.** `scrape_houses()` always starts from `start_page=1`. Re-running
  re-fetches listings already on disk (it overwrites the same path, so it is
  not corrupting, just slow). To skip pages, call
  `scrape_houses(num_pages, start_page=N)` directly rather than via `main.py`.

## Browser backend

`setup_driver()` tries **undetected-chromedriver** first, then falls back to
**plain Selenium**, verifying each by actually loading a page before returning.

This matters: **UC 3.5.5 (the current latest release) is broken against Chrome
157.** The session starts, then every navigation dies with:

```
no such window: target window already closed
from unknown error: web view not found
```

The scraper detects this, prints `undetected-chromedriver unavailable (...);
falling back to Selenium.`, and continues on Selenium with a real user-agent —
which drives Chrome 157 fine and was **not** challenged by DataDome in testing
(a live listing returned its full `window.classified` payload, no captcha).

Force a backend explicitly if needed:

```bash
SCRAPER_DRIVER=selenium .venv/bin/python -u src/main.py 20   # skip UC entirely
SCRAPER_DRIVER=uc       .venv/bin/python -u src/main.py 20   # fail loudly instead of falling back
```

Revisit this once UC ships a release supporting Chrome 157+; UC's whole point is
getting past bot detection, so Selenium is the weaker backend for that.

## Running it unattended

Never hold a foreground shell open for a multi-hour scrape. Use a background
job that logs to a file and wakes you on exit:

```bash
# from the project root
job start scrape-full '.venv/bin/python -u src/main.py 20'
job poll scrape-full        # state, elapsed, idle
job logs scrape-full 40     # tail the log
job stop scrape-full
```

`-u` matters: without it Python block-buffers stdout and the log stays empty
for long stretches, which makes a healthy run look dead.

### Watch progress

Progress is written as OS files, so you can check it without the log at all:

```bash
# listings saved today
find data -name "full_source_$(date +%d%m%Y).html" | wc -l

# files written in the last 10 minutes
find data -maxdepth 1 -type d -mmin -10 | wc -l
```

### Before relaunching after a crash

A previous run that was killed can leave orphaned Chrome/chromedriver
processes behind. These cause the next run to die immediately with:

```
Message: no such window: target window already closed
```

Clean up first:

```bash
pkill -f chromedriver; pkill -f undetected_chromedriver
```

## Cron / "only if online"

`src/run_script_if_internet.sh` wraps a run: it checks connectivity, rotates
`src/run_script.log` past 20 MB, and runs `src/main.py 100`.

All paths are derived from the script's own location, and it no longer
overrides `HOME`, so it works from any checkout. To change how many pages the
cron run covers, edit the `PYTHON_COMMAND` line in that script.

## Parsing only today's files

After an incremental scrape, avoid the full 51 GB re-parse:

```bash
# 1. set TODAY_TAG in src/parse_today_and_append.py to today as DDMMYYYY
#    (it is hardcoded, e.g. '23022026' — this is a real footgun)
# 2. run it
.venv/bin/python -u src/parse_today_and_append.py
```

It appends to `outputs/property_data_parsed.csv` and replaces any existing
rows whose `date_obtained` matches `TODAY_TAG`.

## Verifying the result

```bash
# files actually landed
find data -name "full_source_$(date +%d%m%Y).html" | wc -l

# raw CSV refreshed
ls -lh outputs/property_data_raw.csv

# sanity-check one parsed listing
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs/property_data_raw.csv', low_memory=False)
print(len(df), 'rows;', 'newest date_obtained:', df['date_obtained'].max())
"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no such window: target window already closed` | UC broken on Chrome 157 | handled by fallback; check the log for `falling back to Selenium` |
| Chrome window closes mid-run | the process exited (finished or crashed) | check `job poll` / log; window lifetime == process lifetime |
| Log stays empty for ages | stdout buffering | pass `-u` |
| `ModuleNotFoundError: undetected_chromedriver` | used `.venv-api` | use `.venv` |
| `3 consecutive blocks. Stopping` | rate-limited / IP flagged | wait hours, then resume with `start_page` |
| `No property links found` | page layout changed, or not actually a results page | inspect saved `debug_no_links_found.html` |
| `captcha-delivery` in page | DataDome challenge | handled; backs off automatically |
| Nothing saved but no error | listings already existed, or all pages blocked | check `debug_*` files |
