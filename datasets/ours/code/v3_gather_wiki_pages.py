from pathlib import Path
import json

from mediawiki import MediaWiki

USER_AGENT = "search-and-aggregate (qtq.the.channel@gmail.com)" 
wikipedia = MediaWiki(user_agent=USER_AGENT)

"""
Sources:
https://en.wikipedia.org/wiki/Wikipedia:Top_25_Report/Records
https://en.wikipedia.org/wiki/Portal:Current_events
https://en.wikipedia.org/wiki/Wikipedia:Contents

Recursively take links and backlinks three times, starting from these seed pages.
"""

F0_YEAR_DATA_FILE = Path("datasets/ours/v3/f0_year_data.txt")
V2_COLLECTED_FILE = Path("datasets/ours/v2/collected_wiki_pages.txt")
V3_COLLECTED_FILE = Path("datasets/ours/v3/collected_wiki_pages.txt")
V3_PROGRESS_FILE = Path("datasets/ours/v3/gather_wiki_pages_progress.json")


def load_f0_seed_pages(file_path: Path) -> list[str]:
    """
    Load Top 25 Report page titles from f0_year_data.txt.
    Ignore blank lines and year header lines like '# 2025'.
    """
    if not file_path.exists():
        return []

    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_page_titles(file_path: Path) -> set[str]:
    """
    Load newline-separated page titles from a text file.
    """
    if not file_path.exists():
        return set()

    return {
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_page_if_new(page: str, seen_pages: set[str], blocked_pages: set[str], output_handle) -> bool:
    """
    Append a page title to the output file immediately if it is new and not blocked.
    """
    page = page.strip()
    if not page or page in seen_pages or page in blocked_pages:
        return False

    seen_pages.add(page)
    output_handle.write(page + "\n")
    output_handle.flush()
    print(f"[COLLECTED {len(seen_pages)}] {page}", flush=True)
    return True


def load_progress(file_path: Path) -> dict:
    """
    Load crawl progress from disk.
    """
    if not file_path.exists():
        return {"completed_seed_pages": []}

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed_seed_pages": []}


def save_progress(file_path: Path, completed_seed_pages: set[str]) -> None:
    """
    Save crawl progress to disk after each completed seed page.
    """
    payload = {
        "completed_seed_pages": sorted(completed_seed_pages),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


seed_pages = [
    "Wikipedia:Top_25_Report/Records",
    "Portal:Current_events",
    "Wikipedia:Contents/Reference",
    "Wikipedia:Contents/Culture and the arts",
    "Wikipedia:Contents/Geography and places",
    "Wikipedia:Contents/Health and fitness",
    "Wikipedia:Contents/History and events",
    "Wikipedia:Contents/Human activities",
    "Wikipedia:Contents/Mathematics and logic",
    "Wikipedia:Contents/Natural and physical sciences",
    "Wikipedia:Contents/People and self",
    "Wikipedia:Contents/Philosophy and thinking",
    "Wikipedia:Contents/Religion and belief systems",
    "Wikipedia:Contents/Society and social sciences"
]

# add more F0 seed pages
seed_pages.extend(load_f0_seed_pages(F0_YEAR_DATA_FILE))
seed_pages = list(dict.fromkeys(seed_pages))

existing_v2_pages = load_page_titles(V2_COLLECTED_FILE)
V3_COLLECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
seen_pages = load_page_titles(V3_COLLECTED_FILE)
progress = load_progress(V3_PROGRESS_FILE)

completed_seed_pages = set(progress.get("completed_seed_pages", []))
excluded_overlap_count = 0
file_mode = "a" if V3_COLLECTED_FILE.exists() else "w"
# force to FLUSH
print(f"taken {len(seen_pages)} saved v3 pages from prior run", flush=True)
print(f"taken {len(completed_seed_pages)} completed seed pages from progress file", flush=True)

with open(V3_COLLECTED_FILE, file_mode, encoding="utf-8") as f:
    # progress check for seed pages
    for page in seed_pages:
        if page in existing_v2_pages:
            excluded_overlap_count += 1
            continue
        append_page_if_new(page, seen_pages, existing_v2_pages, f)

    # do 1 iteration ONLY
    for i in range(1):
        pages_to_expand = list(seed_pages)
        print(f"Starting iteration {i+1} with {len(pages_to_expand)} seed pages.", flush=True)
        for idx, page in enumerate(pages_to_expand, 1):
            if page in completed_seed_pages:
                print(f"[SKIP COMPLETED {idx}/{len(pages_to_expand)}] {page}", flush=True)
                continue
            print(f"[FETCH {idx}/{len(pages_to_expand)}] {page}", flush=True)
            try:
                wiki_page = wikipedia.page(page)
                for linked_page in wiki_page.links:
                    if linked_page in existing_v2_pages:
                        excluded_overlap_count += 1
                        continue
                    append_page_if_new(linked_page, seen_pages, existing_v2_pages, f)
                for backlink_page in wiki_page.backlinks:
                    if backlink_page in existing_v2_pages:
                        excluded_overlap_count += 1
                        continue
                    append_page_if_new(backlink_page, seen_pages, existing_v2_pages, f)
                completed_seed_pages.add(page)
                save_progress(V3_PROGRESS_FILE, completed_seed_pages)
                print(f"[CHECKPOINT] Saved progress after '{page}'", flush=True)
            except Exception as e:
                print(f"Error fetching page '{page}': {e}", flush=True)
        # force to FLUSH
        print(f"Iteration {i+1}: {len(seen_pages)} v3-only pages collected so far.", flush=True)

print(f"Excluded {excluded_overlap_count} overlapping page hits from v2 output.", flush=True)
print(f"Saved collected page titles to '{V3_COLLECTED_FILE}'", flush=True)
print(f"Saved progress to '{V3_PROGRESS_FILE}'", flush=True)
