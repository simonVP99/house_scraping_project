import os
import time
import random
import re
import subprocess
from datetime import datetime

from bs4 import BeautifulSoup

try:
    import undetected_chromedriver as uc
except Exception:  # pragma: no cover - optional dependency
    uc = None

# --- Constants ---
BASE_URL = "https://www.immoweb.be/en/search/house-and-apartment/for-sale/gent/9000?countries=BE&isNewlyBuilt=false&page="
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_chrome_major_version():
    """Detect installed Chrome major version on macOS."""
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_path):
        try:
            res = subprocess.run([chrome_path, "--version"], capture_output=True, text=True, check=True)
            match = re.search(r'(\d+)\.', res.stdout)
            if match:
                return int(match.group(1))
        except Exception as e:
            print(f"Failed to detect Chrome version automatically: {e}")
    return None


def setup_driver():
    """
    Initializes a Chrome browser to bypass DataDome CAPTCHA.
    Runs in non-headless mode since headless is detected by DataDome.

    Prefers undetected-chromedriver, but falls back to plain Selenium when UC
    cannot drive the installed Chrome. UC 3.5.5 (the latest release) is broken
    against Chrome 157: the session starts, then every navigation dies with
    "no such window: target window already closed". Plain Selenium with a real
    user-agent drives Chrome 157 fine and is not challenged by DataDome, so the
    fallback is validated by actually loading a page before it is returned.

    Force one backend with SCRAPER_DRIVER=uc or SCRAPER_DRIVER=selenium.
    """
    chrome_version = get_chrome_major_version()
    preference = os.environ.get("SCRAPER_DRIVER", "auto").strip().lower()

    if preference not in ("selenium",) and uc is not None:
        try:
            driver = _build_uc_driver(chrome_version)
            _verify_driver(driver)
            print(f"Undetected Chrome browser initialized (version_main={chrome_version}).")
            return driver
        except Exception as e:
            print(f"undetected-chromedriver unavailable ({type(e).__name__}: {e}); falling back to Selenium.")
            _safe_quit("driver" in dir() and driver or None)

    if preference == "uc":
        raise RuntimeError("SCRAPER_DRIVER=uc requested but undetected-chromedriver is not usable")

    driver = _build_selenium_driver()
    _verify_driver(driver)
    print(f"Selenium Chrome browser initialized (version_main={chrome_version}).")
    return driver


def _build_uc_driver(chrome_version):
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    if chrome_version:
        return uc.Chrome(options=options, version_main=chrome_version)
    return uc.Chrome(options=options)


def _build_selenium_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import tempfile

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Fresh profile per run so a locked/occupied default profile cannot interfere.
    options.add_argument(f"--user-data-dir={tempfile.mkdtemp(prefix='immoscrape_')}")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36"
    )
    # Keep the window visible (non-headless) so DataDome is not triggered.
    return webdriver.Chrome(options=options)


def _verify_driver(driver, timeout=60):
    """Load a cheap page and confirm the browser can actually navigate."""
    deadline = time.time() + timeout
    while True:
        try:
            driver.get("https://www.google.com")
            if driver.page_source:
                return True
        except Exception as e:
            if time.time() > deadline:
                raise RuntimeError(f"browser could not load a page: {e}") from e
            time.sleep(3)


def _safe_quit(driver):
    try:
        if driver is not None:
            driver.quit()
    except Exception:
        pass


def parse_page(html):
    """Parses the search results page to find individual property URLs."""
    soup = BeautifulSoup(html, 'html.parser')
    # Find all links pointing to classified property pages
    links = [a.get('href') for a in soup.find_all('a', href=True)
             if '/classified/' in a.get('href', '')]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


def save_house_data(driver, house_url):
    """Fetches and saves the HTML for a single property page."""
    today = datetime.today().strftime('%d%m%Y')
    try:
        print(f"  -> Navigating to property: {house_url}")
        driver.get(house_url)
        # Polite wait for dynamic content to load
        time.sleep(random.uniform(4, 7))

        html_content = driver.page_source

        # Check if we got blocked on the individual page
        if 'captcha-delivery' in html_content or len(html_content) < 2000:
            print(f"  -> WARNING: Possible CAPTCHA on property page, skipping.")
            return False

        property_id = house_url.split('/')[-1]
        directory = os.path.join(BASE_DIR, 'data', property_id)
        os.makedirs(directory, exist_ok=True)

        file_path = os.path.join(directory, f"full_source_{today}.html")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(html_content)

        print(f"  -> Saved source for property {property_id}")
        return True

    except Exception as e:
        print(f"  -> Error fetching {house_url}: {e}")
        return False


def scrape_houses(num_pages, start_page=1):
    """
    Main scraping function with polite delays to avoid triggering bot detection.
    Uses undetected-chromedriver in non-headless mode to bypass DataDome.
    """
    driver = setup_driver()
    total_saved = 0
    consecutive_failures = 0

    try:
        page = start_page
        while page <= num_pages:
            url = BASE_URL + str(page)
            print(f"\n--- Scraping search results page {page}/{num_pages} ---")

            try:
                driver.get(url)
                # Polite wait for search results to load
                time.sleep(random.uniform(8, 12))

                html_content = driver.page_source

                # Check if blocked by CAPTCHA
                if 'captcha-delivery' in html_content or len(html_content) < 2000:
                    print(f"BLOCKED by DataDome CAPTCHA on page {page}.")
                    with open(os.path.join(BASE_DIR, "debug_blocked_page.html"), "w", encoding="utf-8") as f:
                        f.write(html_content)
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("3 consecutive blocks. Stopping to avoid IP ban.")
                        break
                    # Back off and retry
                    print(f"  Backing off for 60-90 seconds before retry...")
                    time.sleep(random.uniform(60, 90))
                    continue

                house_urls = parse_page(html_content)
                if not house_urls:
                    print("No property links found on this page. Ending scrape.")
                    with open(os.path.join(BASE_DIR, "debug_no_links_found.html"), "w", encoding="utf-8") as f:
                        f.write(html_content)
                    break

                consecutive_failures = 0
                print(f"Found {len(house_urls)} properties. Fetching individual pages...")

                for i, house_url in enumerate(house_urls):
                    success = save_house_data(driver, house_url)
                    if success:
                        total_saved += 1

                    # Polite delay between property pages (5-10 seconds)
                    delay = random.uniform(5, 10)
                    if (i + 1) % 10 == 0:
                        # Extra pause every 10 properties
                        delay = random.uniform(15, 25)
                        print(f"  -- Polite pause after {i+1} properties --")
                    time.sleep(delay)

                # Polite delay between search pages (10-20 seconds)
                between_pages_delay = random.uniform(10, 20)
                print(f"  Waiting {between_pages_delay:.0f}s before next page...")
                time.sleep(between_pages_delay)

                page += 1

            except Exception as e:
                print(f"Error on search page {page}: {e}")
                try:
                    driver.save_screenshot(os.path.join(BASE_DIR, 'debug_error_screenshot.png'))
                except Exception:
                    pass
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("3 consecutive errors. Stopping.")
                    break
                time.sleep(random.uniform(30, 60))

    finally:
        print(f"\nScraping complete. Total properties saved: {total_saved}")
        print("Closing browser.")
        driver.quit()
