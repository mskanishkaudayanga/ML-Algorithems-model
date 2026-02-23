"""
Web Scraper — ikman.lk Mobile Phone Listings
================================================
Sri Lanka Mobile Phone Price Prediction

Ethical Scraping Note:
    This scraper is intended for EDUCATIONAL and RESEARCH purposes only.
    It follows responsible scraping practices:
        - Respects rate limits with configurable delays between requests
        - Mimics a normal browser User-Agent
        - Stops after the configured page limit
        - Does not overload the server with concurrent requests
    Always check and respect the website's robots.txt before scraping.
    If the website's terms of service prohibit scraping, do NOT use this script.

Usage:
    python scrape.py
    python scrape.py --max_pages 200 --output data/mobile_phones.csv
"""

import os
import re
import csv
import json
import time
import random
import logging
import argparse
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://ikman.lk"
LISTING_URL = "https://ikman.lk/en/ads/sri-lanka/mobile-phones"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rate limiting — configurable delays to be respectful to the server
MIN_DELAY = 1.5   # seconds between listing page requests
MAX_DELAY = 3.0
MAX_RETRIES = 3

# CSV columns
CSV_FIELDS = [
    "id", "title", "brand", "model", "price", "price_raw",
    "condition", "storage", "ram", "location",
    "ad_url", "is_member", "is_featured", "scraped_at",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clean_price(price_text):
    """Extract numeric price from text like 'Rs 119,999'."""
    if not price_text:
        return None
    cleaned = re.sub(r"[^\d.]", "", price_text.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def extract_condition(title):
    """Extract condition from title."""
    if not title:
        return "Unknown"
    if "brand new" in title.lower():
        return "Brand New"
    elif "used" in title.lower():
        return "Used"
    return "Unknown"


def extract_brand_model(title):
    """Extract brand and model from the ad title."""
    if not title:
        return "Unknown", "Unknown"

    clean_title = re.sub(r"\s*\((Used|Brand New)\)\s*$", "", title, flags=re.IGNORECASE).strip()

    brands = {
        "Apple": r"^Apple\s+", "Samsung": r"^Samsung\s+",
        "Xiaomi": r"^Xiaomi\s+", "Google": r"^Google\s+",
        "Vivo": r"^Vivo\s+", "Oppo": r"^Oppo\s+",
        "Realme": r"^Realme\s+", "OnePlus": r"^OnePlus\s+",
        "Huawei": r"^Huawei\s+", "Nokia": r"^Nokia\s+",
        "Sony": r"^Sony\s+", "Infinix": r"^Infinix\s+",
        "Tecno": r"^Tecno\s+", "Honor": r"^Honor\s+",
        "ZTE": r"^ZTE\s+", "Nothing": r"^Nothing\s+",
        "Poco": r"^Poco\s+", "LG": r"^LG\s+",
        "Motorola": r"^Motorola\s+", "Redmi": r"^Redmi\s+",
    }
    for brand_name, pattern in brands.items():
        if re.search(pattern, clean_title, re.IGNORECASE):
            model = re.sub(pattern, "", clean_title, flags=re.IGNORECASE).strip()
            return brand_name, model
    return "Other", clean_title


def extract_storage(title):
    """Extract storage capacity from title."""
    if not title:
        return None
    match = re.search(r"(\d+)\s*GB", title, re.IGNORECASE)
    return f"{match.group(1)}GB" if match else None


def extract_ram(title):
    """Extract RAM from title."""
    if not title:
        return None
    match = re.search(r"(\d+)\s*GB\s*[/|]\s*\d+\s*GB", title, re.IGNORECASE)
    if match:
        return f"{match.group(1)}GB"
    match = re.search(r"(\d+)\s*GB\s*Ram", title, re.IGNORECASE)
    return f"{match.group(1)}GB" if match else None


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def safe_request(session, url):
    """Make an HTTP request with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp
            logger.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt})")
            time.sleep(5 * attempt)
        except requests.RequestException as e:
            logger.warning(f"Request error: {e} (attempt {attempt})")
            time.sleep(5 * attempt)
    return None


def scrape_listing_page(session, page_num):
    """Scrape a single listing page for ad data."""
    url = f"{LISTING_URL}?page={page_num}" if page_num > 1 else LISTING_URL
    logger.info(f"Scraping page {page_num}: {url}")

    resp = safe_request(session, url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    ads = []

    # Parse ad links
    ad_links = soup.find_all("a", href=re.compile(r"/en/ad/"))
    seen = set()

    for link in ad_links:
        href = link.get("href", "")
        if not href or href in seen or "/promote" in href or "login-modal" in href:
            continue

        full_url = urljoin(BASE_URL, href)
        if "/en/ad/" not in full_url:
            continue
        seen.add(href)

        link_text = link.get_text(" ", strip=True)
        heading = link.find(["h2", "h3", "h4"])
        title = heading.get_text(strip=True) if heading else link_text
        if not title or len(title) < 3:
            continue

        price_raw = ""
        price_match = re.search(r"Rs\s*[\d,]+", link_text)
        if price_match:
            price_raw = price_match.group(0)

        # Location detection
        location = ""
        sri_lankan_districts = [
            "Colombo", "Gampaha", "Kandy", "Galle", "Matara",
            "Kurunegala", "Ratnapura", "Kalutara", "Badulla", "Ampara",
            "Anuradhapura", "Batticaloa", "Hambantota", "Jaffna",
            "Kegalle", "Kilinochchi", "Mannar", "Matale",
            "Monaragala", "Mullaitivu", "Nuwara Eliya", "Polonnaruwa",
            "Puttalam", "Trincomalee", "Vavuniya",
        ]
        for loc in sri_lankan_districts:
            if loc.lower() in link_text.lower():
                location = loc
                break

        title_clean = title.split("MEMBER")[0].split("FEATURED")[0].strip()
        if title_clean:
            title = title_clean

        brand, model = extract_brand_model(title)

        ads.append({
            "title": title.strip(),
            "brand": brand,
            "model": model,
            "price": clean_price(price_raw),
            "price_raw": price_raw,
            "condition": extract_condition(title),
            "storage": extract_storage(title),
            "ram": extract_ram(title),
            "location": location,
            "ad_url": full_url,
            "is_member": "MEMBER" in link_text,
            "is_featured": "FEATURED" in link_text,
        })

    logger.info(f"  Found {len(ads)} ads on page {page_num}")
    return ads


def run_scraper(max_pages: int, output_csv: str, target: int = 5000):
    """
    Main scraping loop.

    Args:
        max_pages: Maximum listing pages to scrape
        output_csv: Path to save the CSV output
        target: Target number of records to collect
    """
    logger.info("=" * 60)
    logger.info("ikman.lk Mobile Phone Scraper")
    logger.info(f"Target: {target} records | Max pages: {max_pages}")
    logger.info(f"Output: {output_csv}")
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    # Check for existing data (resume support)
    existing_urls = set()
    if os.path.exists(output_csv):
        import pandas as pd
        try:
            df = pd.read_csv(output_csv)
            existing_urls = set(df["ad_url"].dropna().unique())
            logger.info(f"Found {len(existing_urls)} existing records (will skip)")
        except Exception:
            pass

    total = len(existing_urls)
    counter = total
    batch = []
    file_exists = os.path.exists(output_csv) and total > 0

    for page in range(1, max_pages + 1):
        if total >= target:
            break

        listings = scrape_listing_page(session, page)
        if not listings:
            continue

        for ad in listings:
            if ad["ad_url"] in existing_urls:
                continue
            existing_urls.add(ad["ad_url"])
            counter += 1
            ad["id"] = counter
            ad["scraped_at"] = datetime.now().isoformat()
            batch.append(ad)
            total += 1

            if len(batch) >= 25:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True
                    writer.writerows(batch)
                batch = []

            if total >= target:
                break

        logger.info(f"  Total collected: {total}/{target}")
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    # Save remaining
    if batch:
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(batch)

    logger.info("=" * 60)
    logger.info(f"SCRAPING COMPLETE — {total} records saved to {output_csv}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape mobile phone listings from ikman.lk")
    parser.add_argument("--max_pages", type=int, default=200, help="Max pages to scrape")
    parser.add_argument("--output", default="data/mobile_phones.csv", help="Output CSV path")
    parser.add_argument("--target", type=int, default=5000, help="Target record count")
    args = parser.parse_args()
    run_scraper(args.max_pages, args.output, args.target)


if __name__ == "__main__":
    main()
