from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


USER_AGENT = "search-and-aggregate (qtq.the.channel@gmail.com)"
START_YEAR = 2013
END_YEAR = 2026
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
OUTPUT_FILE = Path(__file__).with_name("f0_year_data.txt")


def fetch_category_report_titles(session: requests.Session, year: int) -> List[str]:
    """
    Fetch all report page titles under a yearly Wikipedia Top 25 Reports category.
    Follows category pagination when present and returns titles in page order.
    """
    category_url = (
        f"{WIKIPEDIA_BASE_URL}/wiki/Category:Wikipedia_Top_25_Reports_in_{year}"
    )
    titles: List[str] = []
    seen_titles = set()
    next_url = category_url

    while next_url:
        response = session.get(next_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        mw_pages = soup.find("div", id="mw-pages")
        if mw_pages is None:
            break

        for link in mw_pages.select("li a"):
            title = (link.get("title") or link.get_text(" ", strip=True)).strip()
            if not title.startswith("Wikipedia:Top 25 Report/"):
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)
            titles.append(title)

        next_link = mw_pages.find("a", string=lambda s: s and "next page" in s.lower())
        next_url = urljoin(WIKIPEDIA_BASE_URL, next_link["href"]) if next_link else None

    return titles


def collect_all_years(start_year: int = START_YEAR, end_year: int = END_YEAR) -> Dict[int, List[str]]:
    """Collect report titles for each year in the inclusive year range."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    year_to_titles: Dict[int, List[str]] = {}
    for year in range(start_year, end_year + 1):
        titles = fetch_category_report_titles(session, year)
        year_to_titles[year] = titles
        print(f"{year}: collected {len(titles)} report titles")

    return year_to_titles


def write_year_data(year_to_titles: Dict[int, List[str]], output_file: Path = OUTPUT_FILE) -> None:
    """
    Save yearly report titles to a text file in a readable grouped format.
    """
    lines: List[str] = []
    for year in sorted(year_to_titles):
        titles = year_to_titles[year]
        lines.append(f"# {year}")
        lines.extend(titles)
        lines.append("")

    output_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved {sum(len(v) for v in year_to_titles.values())} titles to {output_file}")


def main() -> None:
    year_to_titles = collect_all_years()
    write_year_data(year_to_titles)


if __name__ == "__main__":
    main()
