"""
ikman.lk Mobile Phone Data Scraper
===================================
Scrapes mobile phone listings from ikman.lk and saves to CSV.
Target: 5,000 records with detailed information.

Usage:
    python src/1_data_collection.py
    python src/1_data_collection.py --pages 200 --output data/mobile_phones.csv
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
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "https://ikman.lk"
LISTING_URL = "https://ikman.lk/en/ads/sri-lanka/mobile-phones"
ADS_PER_PAGE = 28
TARGET_RECORDS = 5000
MAX_PAGES = 200  # ~200 pages × 28 ads = 5,600 ads (buffer for duplicates)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
}

# Rate limiting (seconds) — be respectful
MIN_DELAY = 1.0
MAX_DELAY = 3.0
DETAIL_DELAY_MIN = 0.5
DETAIL_DELAY_MAX = 1.5
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Helper Functions ────────────────────────────────────────────────────────


def setup_session():
    """Create a requests session with retry configuration."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def safe_request(session, url, retries=MAX_RETRIES):
    """Make an HTTP GET request with retries and error handling."""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait = RETRY_BACKOFF * attempt * 2
                logger.warning(f"Rate limited (429). Waiting {wait}s... (attempt {attempt}/{retries})")
                time.sleep(wait)
            elif response.status_code == 403:
                logger.warning(f"Forbidden (403) for {url}. Rotating delay...")
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                logger.warning(f"HTTP {response.status_code} for {url} (attempt {attempt}/{retries})")
                time.sleep(RETRY_BACKOFF)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e} (attempt {attempt}/{retries})")
            time.sleep(RETRY_BACKOFF * attempt)
    logger.error(f"Failed to fetch {url} after {retries} attempts.")
    return None


def clean_price(price_text):
    """Extract numeric price from text like 'Rs 119,999'."""
    if not price_text:
        return None
    # Remove 'Rs', commas, spaces and non-numeric chars
    cleaned = re.sub(r"[^\d.]", "", price_text.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def extract_condition(title):
    """Extract condition (Used/Brand New) from title."""
    title_lower = title.lower() if title else ""
    if "brand new" in title_lower:
        return "Brand New"
    elif "used" in title_lower:
        return "Used"
    return "Unknown"


def extract_brand_model(title):
    """Extract brand and model from the ad title."""
    if not title:
        return "Unknown", "Unknown"

    # Clean the title - remove condition tags
    clean_title = re.sub(r"\s*\((Used|Brand New)\)\s*$", "", title, flags=re.IGNORECASE).strip()

    # Known brands mapping
    brand_patterns = {
        "Apple": r"^Apple\s+",
        "Samsung": r"^Samsung\s+",
        "Xiaomi": r"^Xiaomi\s+",
        "Redmi": r"^Redmi\s+",
        "Google": r"^Google\s+",
        "Vivo": r"^Vivo\s+",
        "Oppo": r"^Oppo\s+",
        "Realme": r"^Realme\s+",
        "OnePlus": r"^OnePlus\s+",
        "Huawei": r"^Huawei\s+",
        "Nokia": r"^Nokia\s+",
        "Sony": r"^Sony\s+",
        "Infinix": r"^Infinix\s+",
        "Tecno": r"^Tecno\s+",
        "Honor": r"^Honor\s+",
        "Motorola": r"^Motorola\s+",
        "ZTE": r"^ZTE\s+",
        "Nothing": r"^Nothing\s+",
        "Poco": r"^Poco\s+",
        "LG": r"^LG\s+",
        "HTC": r"^HTC\s+",
    }

    brand = "Other"
    model = clean_title

    for brand_name, pattern in brand_patterns.items():
        if re.search(pattern, clean_title, re.IGNORECASE):
            brand = brand_name
            model = re.sub(pattern, "", clean_title, flags=re.IGNORECASE).strip()
            break

    return brand, model


def extract_storage(title):
    """Extract storage capacity from title (e.g., '128GB')."""
    if not title:
        return None
    match = re.search(r"(\d+)\s*GB", title, re.IGNORECASE)
    return f"{match.group(1)}GB" if match else None


def extract_ram(title):
    """Extract RAM from title (e.g., '8GB' before storage mention or explicit RAM)."""
    if not title:
        return None
    # Look for patterns like "8GB/128GB" or "8GB 128GB" — first number is usually RAM
    match = re.search(r"(\d+)\s*GB\s*[/|]\s*\d+\s*GB", title, re.IGNORECASE)
    if match:
        return f"{match.group(1)}GB"
    # Look for explicit RAM mentions
    match = re.search(r"(\d+)\s*GB\s*Ram", title, re.IGNORECASE)
    if match:
        return f"{match.group(1)}GB"
    return None


# ─── Page Scraping Functions ─────────────────────────────────────────────────


def scrape_listing_page(session, page_num):
    """
    Scrape a single listing page and return basic ad data.
    
    Returns a list of dicts with: title, price, price_raw, location, 
    condition, ad_url, is_member, is_featured
    """
    url = f"{LISTING_URL}?page={page_num}" if page_num > 1 else LISTING_URL
    logger.info(f"📄 Scraping listing page {page_num}: {url}")

    response = safe_request(session, url)
    if not response:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    ads = []

    # ikman.lk uses Next.js — data is often in __NEXT_DATA__ script tag
    next_data = extract_next_data(soup)
    if next_data:
        ads = parse_next_data_listings(next_data)
        if ads:
            logger.info(f"   ✅ Extracted {len(ads)} ads from __NEXT_DATA__")
            return ads

    # Fallback: parse HTML directly
    ads = parse_html_listings(soup)
    logger.info(f"   ✅ Extracted {len(ads)} ads from HTML")
    return ads


def extract_next_data(soup):
    """Extract JSON data from Next.js __NEXT_DATA__ script tag."""
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if script_tag and script_tag.string:
        try:
            return json.loads(script_tag.string)
        except json.JSONDecodeError:
            logger.debug("Failed to parse __NEXT_DATA__")
    return None


def parse_next_data_listings(next_data):
    """Parse ad listings from __NEXT_DATA__ JSON structure."""
    ads = []
    try:
        # Navigate to the ads data — structure varies, try common paths
        page_props = next_data.get("props", {}).get("pageProps", {})

        # Try different possible keys
        ads_data = (
            page_props.get("ads", [])
            or page_props.get("adList", [])
            or page_props.get("data", {}).get("ads", [])
            or page_props.get("initialData", {}).get("ads", [])
        )

        # Also check for paginatedAds or serp structure
        if not ads_data:
            serp = page_props.get("serp", {})
            ads_data = serp.get("ads", [])

        if not ads_data:
            # Try deeper nested structures
            for key, value in page_props.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list) and len(sub_value) > 0:
                            if isinstance(sub_value[0], dict) and any(
                                k in sub_value[0] for k in ["title", "name", "price", "slug"]
                            ):
                                ads_data = sub_value
                                break
                    if ads_data:
                        break

        for ad in ads_data:
            if not isinstance(ad, dict):
                continue

            title = ad.get("title") or ad.get("name") or ad.get("subject", "")
            if not title:
                continue

            # Extract price
            price_raw = ad.get("price", "")
            if isinstance(price_raw, dict):
                price_raw = price_raw.get("value", "") or price_raw.get("amount", "")
            price = clean_price(str(price_raw))

            # Extract location
            location = ""
            loc_data = ad.get("location", ad.get("area", ""))
            if isinstance(loc_data, dict):
                location = loc_data.get("name", "") or loc_data.get("city", "")
            elif isinstance(loc_data, str):
                location = loc_data

            # Extract ad URL
            slug = ad.get("slug", "") or ad.get("url", "")
            ad_url = f"{BASE_URL}/en/ad/{slug}" if slug and not slug.startswith("http") else slug

            # Check for membership/featured status
            is_member = ad.get("isMember", False) or ad.get("membership", False)
            is_featured = ad.get("isFeatured", False) or ad.get("featured", False)

            condition = extract_condition(title)
            brand, model = extract_brand_model(title)
            storage = extract_storage(title)
            ram = extract_ram(title)

            ads.append({
                "title": title.strip(),
                "price": price,
                "price_raw": str(price_raw).strip(),
                "location": location.strip(),
                "condition": condition,
                "brand": brand,
                "model": model,
                "storage": storage,
                "ram": ram,
                "ad_url": ad_url,
                "is_member": is_member,
                "is_featured": is_featured,
            })

    except Exception as e:
        logger.debug(f"Error parsing __NEXT_DATA__: {e}")

    return ads


def parse_html_listings(soup):
    """Parse ad listings from HTML when __NEXT_DATA__ is not available."""
    ads = []

    # Find all ad links — ikman uses anchor tags with ad detail URLs
    ad_links = soup.find_all("a", href=re.compile(r"/en/ad/"))

    # Track seen URLs to avoid duplicates on same page
    seen_urls = set()

    for link in ad_links:
        href = link.get("href", "")
        if not href or href in seen_urls:
            continue

        # Only process listing ad links (not boost/promote links)
        if "/promote" in href or "login-modal" in href or "/boost" in href:
            continue

        full_url = urljoin(BASE_URL, href)

        # Skip if not a proper ad URL
        if "/en/ad/" not in full_url:
            continue

        seen_urls.add(href)

        # Try to extract title from link text or nested elements
        title = ""
        # Look for h2 or heading-like elements
        heading = link.find(["h2", "h3", "h4"])
        if heading:
            title = heading.get_text(strip=True)
        else:
            title = link.get_text(strip=True)

        if not title or len(title) < 3:
            continue

        # Try to extract price from nearby elements
        price_raw = ""
        price = None

        # Look for price pattern in the text content
        link_text = link.get_text(" ", strip=True)
        price_match = re.search(r"Rs\s*[\d,]+", link_text)
        if price_match:
            price_raw = price_match.group(0)
            price = clean_price(price_raw)

        # Extract location from the text
        location = ""
        # Common locations in Sri Lanka
        locations = [
            "Colombo", "Gampaha", "Kandy", "Galle", "Matara",
            "Kurunegala", "Ratnapura", "Kalutara", "Badulla", "Ampara",
            "Anuradhapura", "Batticaloa", "Hambantota", "Jaffna",
            "Kegalle", "Kilinochchi", "Mannar", "Matale",
            "Monaragala", "Mullaitivu", "Nuwara Eliya", "Polonnaruwa",
            "Puttalam", "Trincomalee", "Vavuniya",
        ]
        for loc in locations:
            if loc.lower() in link_text.lower():
                location = loc
                break

        # Check for MEMBER badge
        is_member = "MEMBER" in link_text
        is_featured = "FEATURED" in link_text

        condition = extract_condition(title)
        brand, model = extract_brand_model(title)
        storage = extract_storage(title)
        ram = extract_ram(title)

        # Clean the title (remove extra text from link)
        # The title is usually the first meaningful text before MEMBER/price
        title_clean = title.split("MEMBER")[0].split("FEATURED")[0].strip()
        if title_clean:
            title = title_clean

        ads.append({
            "title": title.strip(),
            "price": price,
            "price_raw": price_raw,
            "location": location,
            "condition": condition,
            "brand": brand,
            "model": model,
            "storage": storage,
            "ram": ram,
            "ad_url": full_url,
            "is_member": is_member,
            "is_featured": is_featured,
        })

    return ads


def scrape_ad_detail(session, ad_url):
    """
    Scrape additional details from an individual ad page.
    
    Returns dict with extra fields: description, seller_name, posted_date, etc.
    """
    if not ad_url or not ad_url.startswith("http"):
        return {}

    response = safe_request(session, ad_url, retries=2)
    if not response:
        return {}

    soup = BeautifulSoup(response.text, "lxml")
    details = {}

    # Try __NEXT_DATA__ first
    next_data = extract_next_data(soup)
    if next_data:
        try:
            page_props = next_data.get("props", {}).get("pageProps", {})
            ad_data = (
                page_props.get("ad", {})
                or page_props.get("adDetail", {})
                or page_props.get("data", {}).get("ad", {})
                or page_props
            )

            if isinstance(ad_data, dict):
                details["description"] = (
                    ad_data.get("description", "")
                    or ad_data.get("body", "")
                    or ""
                )[:500]  # Truncate long descriptions

                # Extract properties/attributes
                props = ad_data.get("properties", []) or ad_data.get("attributes", [])
                if isinstance(props, list):
                    for prop in props:
                        if isinstance(prop, dict):
                            key = prop.get("key", "") or prop.get("name", "")
                            value = prop.get("value", "") or prop.get("label", "")
                            key_lower = key.lower()
                            if "brand" in key_lower:
                                details["detail_brand"] = str(value)
                            elif "model" in key_lower:
                                details["detail_model"] = str(value)
                            elif "condition" in key_lower:
                                details["detail_condition"] = str(value)

                # Seller info
                seller = ad_data.get("seller", {}) or ad_data.get("user", {})
                if isinstance(seller, dict):
                    details["seller_name"] = seller.get("name", "") or seller.get("displayName", "")

                # Date
                details["posted_date"] = ad_data.get("date", "") or ad_data.get("createdAt", "")

        except Exception as e:
            logger.debug(f"Error parsing ad detail __NEXT_DATA__: {e}")

    # Fallback: parse HTML for description
    if "description" not in details:
        desc_div = soup.find("div", class_=re.compile(r"description", re.I))
        if desc_div:
            details["description"] = desc_div.get_text(strip=True)[:500]
        else:
            # Try to get the main content area
            main_content = soup.find("main") or soup.find("article")
            if main_content:
                text = main_content.get_text(" ", strip=True)
                # Take a reasonable excerpt
                details["description"] = text[:500] if len(text) > 500 else text

    return details


# ─── CSV Management ──────────────────────────────────────────────────────────

CSV_FIELDS = [
    "id", "title", "brand", "model", "price", "price_raw",
    "condition", "storage", "ram", "location",
    "description", "detail_brand", "detail_model", "detail_condition",
    "seller_name", "posted_date",
    "ad_url", "is_member", "is_featured",
    "scraped_at",
]


def get_existing_urls(csv_path):
    """Load already-scraped ad URLs from existing CSV to avoid duplicates."""
    urls = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "ad_url" in df.columns:
                urls = set(df["ad_url"].dropna().unique())
            logger.info(f"📂 Found {len(urls)} existing records in {csv_path}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    return urls


def append_to_csv(csv_path, records):
    """Append records to the CSV file."""
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


# ─── Main Scraping Pipeline ─────────────────────────────────────────────────


def run_scraper(output_path, max_pages=MAX_PAGES, target=TARGET_RECORDS, scrape_details=True):
    """
    Main scraping pipeline.
    
    Args:
        output_path: Path to save CSV file
        max_pages: Maximum listing pages to scrape
        target: Target number of records
        scrape_details: Whether to visit individual ad pages for more info
    """
    logger.info("=" * 70)
    logger.info("🚀 ikman.lk Mobile Phone Scraper — Starting")
    logger.info(f"   Target: {target} records | Max pages: {max_pages}")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Detail scraping: {'ON' if scrape_details else 'OFF'}")
    logger.info("=" * 70)

    # Setup
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    session = setup_session()
    existing_urls = get_existing_urls(output_path)
    total_scraped = len(existing_urls)
    record_counter = total_scraped
    batch_buffer = []
    BATCH_SIZE = 25  # Save every 25 records

    start_time = time.time()
    pages_with_no_new_ads = 0

    for page_num in range(1, max_pages + 1):
        if total_scraped >= target:
            logger.info(f"🎯 Reached target of {target} records!")
            break

        # Scrape listing page
        listings = scrape_listing_page(session, page_num)

        if not listings:
            pages_with_no_new_ads += 1
            if pages_with_no_new_ads >= 5:
                logger.warning("⚠️ No new ads found for 5 consecutive pages. Stopping.")
                break
            logger.warning(f"   ⚠️ No ads found on page {page_num}")
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            continue

        new_ads = 0
        for listing in listings:
            ad_url = listing.get("ad_url", "")
            if ad_url in existing_urls:
                continue

            existing_urls.add(ad_url)
            record_counter += 1

            # Add metadata
            listing["id"] = record_counter
            listing["scraped_at"] = datetime.now().isoformat()

            # Optionally scrape detail page
            if scrape_details and ad_url:
                detail_data = scrape_ad_detail(session, ad_url)
                listing.update(detail_data)
                time.sleep(random.uniform(DETAIL_DELAY_MIN, DETAIL_DELAY_MAX))

            batch_buffer.append(listing)
            new_ads += 1
            total_scraped += 1

            # Save batch
            if len(batch_buffer) >= BATCH_SIZE:
                append_to_csv(output_path, batch_buffer)
                logger.info(
                    f"   💾 Saved batch ({len(batch_buffer)} records). "
                    f"Total: {total_scraped}/{target}"
                )
                batch_buffer = []

            if total_scraped >= target:
                break

        if new_ads > 0:
            pages_with_no_new_ads = 0
        else:
            pages_with_no_new_ads += 1

        # Progress report
        elapsed = time.time() - start_time
        rate = total_scraped / max(elapsed / 60, 0.01)
        eta_min = (target - total_scraped) / max(rate, 0.01)
        logger.info(
            f"   📊 Page {page_num}: +{new_ads} new ads | "
            f"Total: {total_scraped}/{target} | "
            f"Rate: {rate:.1f}/min | ETA: {eta_min:.0f}min"
        )

        # Rate limiting between pages
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    # Save any remaining records
    if batch_buffer:
        append_to_csv(output_path, batch_buffer)
        logger.info(f"   💾 Saved final batch ({len(batch_buffer)} records)")

    # Final report
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Scraping Complete!")
    logger.info(f"   Total records: {total_scraped}")
    logger.info(f"   Time elapsed: {elapsed / 60:.1f} minutes")
    logger.info(f"   Output file: {output_path}")
    logger.info("=" * 70)

    return total_scraped


# ─── Quick Scraper (Listing Pages Only — No Detail Pages) ────────────────────


def run_quick_scraper(output_path, max_pages=MAX_PAGES, target=TARGET_RECORDS):
    """
    Fast scraper that only scrapes listing pages (no detail page visits).
    Much faster but gets less detail per record.
    """
    return run_scraper(output_path, max_pages, target, scrape_details=False)


# ─── Entry Point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Scrape mobile phone data from ikman.lk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/1_data_collection.py
  python src/1_data_collection.py --pages 200 --target 5000
  python src/1_data_collection.py --quick --target 5000
  python src/1_data_collection.py --output data/phones.csv
        """,
    )
    parser.add_argument(
        "--output", "-o",
        default="data/mobile_phones.csv",
        help="Output CSV file path (default: data/mobile_phones.csv)",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=MAX_PAGES,
        help=f"Maximum number of listing pages to scrape (default: {MAX_PAGES})",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=TARGET_RECORDS,
        help=f"Target number of records (default: {TARGET_RECORDS})",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: scrape listing pages only (no detail pages, much faster)",
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_scraper(args.output, args.pages, args.target)
    else:
        run_scraper(args.output, args.pages, args.target)


if __name__ == "__main__":
    main()
