import os
import random
import argparse
import json
import pickle

import requests
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import unquote
from mediawiki import MediaWiki

from wikipedia_table_processing import process_table
from wikipedia_infobox_extraction import parse_infobox_table
from helper import *
from dspy_program_qar_gen import *

USER_AGENT = "search-and-aggregate (qtq.the.channel@gmail.com)" # for our project.
wikipedia = MediaWiki(user_agent=USER_AGENT)


# -------------------------------------------------------
# Constants / local config objects (edit as needed)
# -------------------------------------------------------

# Minimum fraction of cells in a column that must have Wikipedia links
# for the column to be considered a "reference column" for Method 1.1
MIN_WIKI_LINK_FRACTION = 1.0  # require ALL cells to have a wiki link

# Wikipedia base URL for constructing full URLs from /wiki/... hrefs
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"

# Prefix that identifies an internal Wikipedia article link
WIKI_ARTICLE_PREFIX = "/wiki/"

# Categories of Wikipedia pages to skip (these are not article pages)
WIKI_NON_ARTICLE_PREFIXES = [
    "/wiki/Special:",
    "/wiki/Help:",
    "/wiki/Wikipedia:",
    "/wiki/Talk:",
    "/wiki/File:",
    "/wiki/Category:",
    "/wiki/Portal:",
    "/wiki/Template:",
]

# Page title-pageid mapping (for caching wikipedia pages with numeric pageids)
# WIKI_TITLE_PAGEID_MAPPING = json.load(open("datasets/wikipedia_title_pageid_mapping.json", "r"))
WIKI_CACHE_DIR = os.path.expanduser("~/.cache/wikipedia/pages/")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

# -------------------------------------------------------
# Helpers for crawling Wikipedia pages and extracting tables + context
# -------------------------------------------------------

def crawl_wikipedia_page(title: str, verbose: bool=False) -> MediaWiki.page:
    """
    Fetch a Wikipedia page by title using the mediawiki library.
    Returns a MediaWiki.page object with attributes like html, summary, url, etc.
    Enable caching to avoid repeated requests during development and testing.
    """
    # pageid = WIKI_TITLE_PAGEID_MAPPING.get(title, title)  # use title if pageid not found
    pageid = title
    cache_file_path = os.path.join(WIKI_CACHE_DIR, f"{pageid.replace('/', '-').replace('\\', '-')}.pkl")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            page = pickle.load(f)
            f.close()
            if verbose:
                print(f"  [CACHE] Loaded page '{title}' ({pageid=}) from cache.")
            return page
    else:
        page = wikipedia.page(title)
        # WIKI_TITLE_PAGEID_MAPPING[title] = page.pageid  # update mapping with new pageid
        with open(cache_file_path, "wb") as f:
            pickle.dump(page, f)
            f.close()
            if verbose:
                print(f"  [CACHE] Saved page '{title}' ({pageid=}) to cache.")
            return page


def wikipedia_url2bs4(url):
    """
    Given a Wikipedia URL, fetch the page content and return a BeautifulSoup object for parsing.
    Have to use requests instead of wikipedia.page(url) because the latter does not support URL input
    """
    response = requests.get(url, headers={'User-Agent': ''})
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


def check_table_regularity(data_df):
    """
    TODO: Checks whether a given table is regular. More criteria to be added.
    
    Criteria for regularity:
    - sum of #rows and #columns are at least 4 (to avoid very small tables)
    - # data_df has no n/a values (or empty values)
    - no column duplicate names
    - values in the same column have the same unit (e.g., all in million, or all in billion)
    - values are easy to understand (e.g., no typo, no weird symbols, etc.)
    """
    conditions = [
        sum(data_df.shape) < 4,
        data_df.isnull().values.any(),
        data_df.columns.duplicated().any(),
    ]
    return not any(conditions)


def check_table_with_ideal_size(df):
    """
    Check if the table has at least/most min_rows/max_rows rows and min_columns/max_columns columns
    to ensure it's not too small and not too large.
    """
    min_rows, max_rows = 5, 30
    min_columns, max_columns = 2, 8
    max_cells = 150  # to avoid too large tables
    n_rows, n_columns = df.shape[0], df.shape[1]
    return (min_rows <= n_rows <= max_rows) and \
        (min_columns <= n_columns <= max_columns) and \
        (n_rows * n_columns <= max_cells)


# -------------------------------------------------------
# Helper: resolve a /wiki/... href to a Wikipedia page title
# -------------------------------------------------------

def wiki_href_to_title(href: str) -> str | None:
    """
    Convert a relative Wikipedia href like '/wiki/Zulia' to a page title 'Zulia'.
    Returns None if the href does not point to a regular article.
    """
    # if None, NaN, or does not start with /wiki/, return None
    if not href or not str(href).startswith(WIKI_ARTICLE_PREFIX):
        return None
    # Skip non-article namespaces
    for prefix in WIKI_NON_ARTICLE_PREFIXES:
        if href.startswith(prefix):
            return None
    # Remove the /wiki/ prefix, decode percent-encoding, replace underscores
    title = href[len(WIKI_ARTICLE_PREFIX):]
    title = unquote(title)          # decode %C3%B3 → ó etc.
    title = title.replace("_", " ") # underscores → spaces
    # Remove any fragment (#...) from anchor links
    title = title.split("#")[0]
    return title if title else None


# -------------------------------------------------------
# Helper: find columns where every row has a Wikipedia link
# -------------------------------------------------------

def find_wikipedia_reference_columns(reference_df, min_fraction: float = MIN_WIKI_LINK_FRACTION):
    """
    Given the reference_df produced by process_table, return a list of
    column names where at least `min_fraction` of the rows contain a Wikipedia
    article link (i.e., an href starting with /wiki/ that is a valid article).

    Returns:
        List of (col_name, col_index) tuples for qualifying columns.
    """
    qualifying = []
    for col_idx, col in enumerate(reference_df.columns):
        n_rows = len(reference_df)
        if n_rows == 0:
            continue
        n_with_wiki_link = 0
        for href in reference_df[col]:
            has_wiki = wiki_href_to_title(href) is not None
            if has_wiki:
                n_with_wiki_link += 1
        fraction = n_with_wiki_link / n_rows
        if fraction >= min_fraction:
            qualifying.append((col, col_idx))
    return qualifying


# -------------------------------------------------------
# Helper: extract Wikipedia article titles from a reference_df column
# -------------------------------------------------------

def extract_wiki_titles_from_ref_column(reference_df, col_name: str) -> list[str | None]:
    """
    For each row in reference_df[col_name], extract the content from the reference URL.
    Returns a list aligned with the rows.
    """
    titles = []
    for href in reference_df[col_name]:
        title = None
        t = wiki_href_to_title(href)
        if t is not None:
            title = t
        titles.append(title)
    return titles


# -------------------------------------------------------
# Helpers: find nearest section title / hatnote for a table
# -------------------------------------------------------

def _find_nearest(table_tag, *, what="section_title") -> str:
    """
    Walk backward through the DOM siblings/ancestors of `table_tag` to find
    the nearest heading or hatnote that precedes it. Returns the (text, [urls]) or ("", []).
    """
    tag_maps = {
        "section_title": "mw-heading",
        "hatnote": "hatnote",
    }
    if what not in tag_maps:
        raise ValueError(f"Unsupported 'what' value: {what}. Supported values are: {list(tag_maps.keys())}")
    else:
        what_class = tag_maps.get(what)
    extract_links = True if what == "hatnote" else False

    if what == "hatnote":
        # Walk backward through siblings to find the nearest hatnote
        for i in range(2):  # check current level and one level up
            sibling = table_tag.find_previous_sibling(class_=what_class)
            if sibling:
                return extract_text(sibling, extract_links=extract_links)
            elif table_tag.parent:
                table_tag = table_tag.parent

    elif what == "section_title":
        # Get more section titles, from the lowest level until the page title.
        all_section_titles = "" # just concatenate all section titles, from highest level to lowest level
                                # e.g., Demographics -> Largest cities
        what_class = "mw-heading"  # the lowest section level we consider (h6)
        for i in range(2):  # check current level and one level up
            sibling = table_tag.find_previous_sibling(class_=what_class)
            if sibling:
                all_section_titles = extract_text(sibling, extract_links=extract_links)
            elif table_tag.parent:
                table_tag = table_tag.parent

        # get section class which is not "mw-heading" but contains "mw-heading" (e.g., "mw-heading2")
        if sibling:
            sibing_classes = [cls for cls in sibling.get("class", []) if "mw-heading" in cls and cls != "mw-heading"]
            if sibing_classes:
                try:
                    level = int(sibing_classes[0].replace("mw-heading", ""))
                    for l in range(level-1, 1, -1):  # check higher levels until h2
                        sibling = table_tag.find_previous_sibling(class_=f"mw-heading{l}")
                        if sibling:
                            title_text = extract_text(sibling, extract_links=extract_links)
                            if title_text:
                                all_section_titles = title_text + " / " + all_section_titles
                except Exception as e:
                    pass
        return all_section_titles.strip(" / ")


# -------------------------------------------------------
# Method 1.1 — Table Expansion via Reference Page Summaries
# -------------------------------------------------------

def method_1_1_expand_table(args, data_df, reference_df, ref_col_name: str, table_metadata: dict):
    """
    Method 1.1: Expand a table by 
        fetching summaries of Wikipedia references for all entries in `ref_col_name` and 
        using an LLM to identify common attributes and fill new columns.

    Pipeline:
        1. Fetch the summary (intro) of each referenced Wikipedia page.
        2. (LLM) Identify common attributes across all summaries, 
            extract per-row values, and add new columns to the table in html format.
        3. Parse and validate the expanded table (format, completeness, info consistency).

    Args:
        args:             Parsed CLI arguments (args.model is used for LLM calls).
        data_df:          pd.DataFrame — the original table data.
        reference_df:     pd.DataFrame — hrefs per cell (from process_table).
        ref_col_name:     Name of the column whose entries link to Wikipedia pages.
        table_metadata:   JSON - metadata of table (caption, other_info_references, section_title, hatnote).

    Returns:
        dict with keys:
            expanded_table_html : str
            new_columns : list of str
            notes : str
    """

    print("=" * 50)
    print(f"\n  [Method 1.1] Expanding table using reference column: '{ref_col_name}'")

    # -------------------------------------------------------
    # Step 1: Fetch Wikipedia summaries for each referenced page
    # -------------------------------------------------------
    # Extract Wikipedia page titles from the reference column
    ref_titles = extract_wiki_titles_from_ref_column(reference_df, ref_col_name)
    
    summaries = {}   # {title: summary_text}
    for i, title in enumerate(ref_titles):
        entry = data_df.loc[i, ref_col_name]
        if title is None:
            continue
        if title in summaries:
            continue  # avoid duplicate fetches
        # print(f"  Step 1: Fetching summary of page '{title}' for entry '{entry}' ...")
        # summaries[entry] = crawl_wikipedia_page(title, args.debug).summary
        # try to take the first infobox in the page first, if fail, fallback to summary
        page = crawl_wikipedia_page(title, args.debug)
        soup = BeautifulSoup(page.html, 'html.parser')
        infobox = soup.find('table', class_='infobox')
        if infobox:
            infobox_data = parse_infobox_table(infobox, verbose=args.debug)
            summaries[entry] = json.dumps(infobox_data, indent=2, ensure_ascii=False)
        else:
            summaries[entry] = page.summary

    # Compose a human-readable block of summaries for LLM input
    summaries_block = "\n\n---\n\n".join(
        f"[{entry}]\n{summary if summary else '(no summary available)'}"
        for entry, summary in summaries.items()
    )

    # -------------------------------------------------------
    # Step 2: Use LLM to identify common attributes and expand the table
    # -------------------------------------------------------
    original_table_html = data_df.to_html(index=False)

    messages = get_messages_templates(TableExpansion)
    messages[-1]["content"] = messages[-1]["content"].format(
        table=original_table_html,
        metadata=json.dumps(table_metadata, indent=2, ensure_ascii=False),
        column_name=ref_col_name,
        wiki_summaries=summaries_block,
    )
    # if args.debug:
    #     print("\n  [DEBUG] LLM input messages:")
    #     for msg in messages:
    #         print(f"  - {msg['role']}:\n{msg['content']}\n")
    response = llm_call(args=args, messages=messages)
    response = TableExpansion_parser(response)
    expanded_table_html = response["enriched_table"]
    expansion_notes = response["notes"]

    try:
        # parse content inside ```html ...```
        # match = re.search(r"```html(.*?)```", expanded_table_html, re.DOTALL)
        # if match:
        #     expanded_table_html = match.group(1).strip()
        print(f"  Step 2: Received expanded table HTML from LLM as follows:\n{expanded_table_html}\n")
        expanded_df = pd.read_html(StringIO(expanded_table_html))[0]
        print(f"  Step 2: Successfully parsed expanded table HTML into DataFrame with shape {expanded_df.shape}.")
        print(f"  Table columns: \n{expanded_df.columns}\n")
        print(f"  Table content: \n{expanded_df.to_markdown(index=False)}\n")
        print(f"  Notes from LLM:\n{expansion_notes}\n")

    except Exception as e:
        print(f"  [ERROR] Could not parse expanded html table: {e}")
        return {}

    # -------------------------------------------------------
    # Step 3: Validate the expanded table
    # -------------------------------------------------------
    try:
        validation_result = validate_expanded_table(expanded_df, data_df, ref_col_name, summaries)
    except Exception as e:
        print(f"  [ERROR] Exception during expanded table validation: {e}")
        validation_result = False

    # If pass the checking,
    # Return the expanded table in html format along with notes about the expansion process
    if validation_result:
        print("  Table expansion successful and validated.")
        result = dict(
            expanded_table_html=expanded_df.to_html(index=False),
            notes=expansion_notes,
            new_columns=[c for c in expanded_df.columns if c not in data_df.columns],
            new_evidence_wikipages=ref_titles
        )
        return result
    else:
        return {}


def validate_expanded_table(expanded_df, data_df, ref_col_name, summaries):
    # Check 0: check regularity (e.g., no n/a values, no duplicate column names, etc.)
    regularity_ok = check_table_regularity(expanded_df)
    print(f"  Step 3a: Expanded table regularity check: {'PASS' if regularity_ok else 'FAIL'}")
    # if not regularity_ok:
    #     return False

    # -------------------------------------------------------
    # Check 1: Ensure value consistency with original table
    #   - All original columns must still be present in expanded_df
    #   - Values in those columns must be identical to data_df
    # -------------------------------------------------------
    original_cols = list(data_df.columns)
    expanded_cols = list(expanded_df.columns)

    for c in original_cols:
        if c not in expanded_df.columns:
            expanded_df[c] = data_df[c] # add missing original column back for consistency check

    value_consistency_ok = True
    for col in original_cols:
        orig_vals = data_df[col].astype(str).str.strip().tolist()
        expd_vals = expanded_df[col].astype(str).str.strip().tolist()

        if orig_vals != expd_vals:
            print(f"  Step 3b: Value consistency check FAIL — column '{col}' was modified by LLM.")
            print(f"    Original : {orig_vals}")
            print(f"    Expanded : {expd_vals}")
            value_consistency_ok = False
            # NOTE: As we prompt LLM to fix possible typo, 
            # we can be tolerant to minor modifications in the original columns.
            # print(f"    Reverted column '{col}' to original values for further checks.")
            # expanded_df[col] = data_df[col] # revert to original values for consistency

    if value_consistency_ok:
        print("  Step 3b: Value consistency with original table: PASS")

    # -------------------------------------------------------
    # Check 2: New column deduplication and non-triviality
    #   - No two columns in expanded_df share the same name
    #   - No two *new* columns have identical value lists
    #   - Each new column should have at least 2 unique non-null values (to avoid trivial/constant columns).
    # -------------------------------------------------------
    # Identify new columns added by the LLM
    new_cols = [c for c in expanded_cols if c not in original_cols]
    print(f"  Step 3b: New columns added by LLM: {new_cols}")
    for col in new_cols:
        unique_non_null_vals = expanded_df[col].dropna().unique()
        if len(unique_non_null_vals) < 2:
            expanded_df.drop(columns=[col], inplace=True)  # drop trivial column for further checks

    # re-identify new columns after dropping trivial ones
    new_cols = [c for c in expanded_df.columns if c not in original_cols]
    if not new_cols:
        print("  Step 3b: No new non-trivial columns were added. Skipping further checks.")
        return False

    if expanded_df.columns.duplicated().any():
        dup_names = expanded_df.columns[expanded_df.columns.duplicated()].tolist()
        print(f"  Step 3c: Column deduplication check FAIL — duplicate column names: {dup_names}")
        return False

    dedup_ok = True
    dup_col_names = []
    new_col_value_map = {}  # col_name -> sorted values in the column (for pairwise comparison)
    for col in new_cols:
        if expanded_df[col].isnull().any():
            continue
        col_vals = sorted(expanded_df[col].astype(str).str.strip().tolist())
        for existing_col, existing_vals in new_col_value_map.items():
            if col_vals == existing_vals:
                print(f"  Step 3c: Column deduplication check FAIL — "
                      f"columns '{col}' and '{existing_col}' have identical values.")
                dedup_ok = False
                dup_col_names.append(col)
                break
        if col not in dup_col_names:
            new_col_value_map[col] = col_vals

    if dedup_ok:
        print("  Step 3c: Column deduplication check: PASS")
    else:
        print(f"  Step 3c: Dropping duplicate columns: {dup_col_names} for further checks.")
        expanded_df.drop(columns=dup_col_names, inplace=True)  # drop duplicate columns for further checks

    # -------------------------------------------------------
    # Check 3: Value consistency of new columns with reference summaries
    #   - For each row, each new-column value should appear as a substring
    #     (case-insensitive) in the summary of the corresponding reference page.
    # -------------------------------------------------------
    ref_entries = data_df[ref_col_name].astype(str).str.strip().tolist()
    summary_check_ok = True

    for col in new_cols:
        for i, entry in enumerate(ref_entries):
            summary = summaries[entry].lower()
            cell_value = expanded_df.loc[i, col]

            # check if the cell_value is None, nan, na
            if pd.isnull(expanded_df.loc[i, col]) or \
                str(cell_value).strip().lower() in ["none", "nan", "na", "n/a", ""]:
                continue

            # preprocessing for edge cases
            # - int mix with nan -> float. Convert back to int.
            if isinstance(cell_value, float) and cell_value.is_integer():
                cell_value = int(cell_value)
            
            # Accept if 
            # - cell value (or any token of it) appears in the summary (case-insensitive)
            # - comma/semicolon-separated multi-value cells if any part matches
            # - "text (note)" format -> check if text and note in wiki_summary
            cell_value = str(cell_value).strip().lower()
            if cell_value not in summary:
                parts = [p.strip() for p in re.split(r'[,\;\)\(]', cell_value) if p.strip()]
                if not all(p in summary for p in parts):
                    print(f"  Step 3d: Value consistency check FAIL — value '{cell_value}' in column '{col}' for entry '{entry}' does not appear in the reference summary.")
                    summary_check_ok = False

    if not summary_check_ok:
        return False
    else:
        print("  Step 3d: New-column value consistency with reference summaries: PASS")

    # -------------------------------------------------------
    # Final check: LLM-based review
    # -------------------------------------------------------

    return True


# -------------------------------------------------------
# Main functions: iterate over a list of page titles and extract/expand tables
# -------------------------------------------------------

def extract_table_and_context(args, title: str, table_ids: list[str]=None):
    """
    Given a Wikipedia page title, extract qualified tables and their surrounding
    context (caption, references, hatnotes, section title), then route to the
    appropriate expansion method.

    Currently only Method 1.1 is implemented; other cases are skipped.

    Args:
        args:           Parsed CLI arguments.
        title:          Wikipedia page title.
        table_ids:      Optional list of table IDs to re-process (if None, process all tables on the page).
    
    Return:
        output_records: List of dicts of expanded tables with keys:
            table_id
            page_metadata (page_title, page_url, revision_id, parent_id, etc.)
            table_metadata (caption, hatnotes, section_title, references, etc.)
            original_table_html
            expanded_table_html
            notes   (notes about expanded table, if any, used for QA generation)
            new_columns  (use for QA generation as a column of interest)
            new_evidence_wikipages (for gathering context for QA)
    """
    output_records = []

    # Fetch the page via mediawiki library
    try:
        page = crawl_wikipedia_page(title, args.debug)
    except Exception as e:
        print(f"[ERROR] Could not fetch page '{title}': {e}")
        return

    print("=" * 80)
    print(f"Title:   {title}")
    print(f"Summary: {page.summary[:200]}...")
    print(f"URL:     {page.url}")
    print("-" * 80)

    # Build page-level metadata (used in every output record)
    page_metadata = {
        "page_title": title,
        "page_summary": page.summary,
        "page_url": page.url,
        "page_id": page.pageid,
        "revision_id": page.revision_id,
        "parent_id": page.parent_id,
    }

    soup = BeautifulSoup(page.html, 'html.parser')

    # Collect all HTML elements to help us find the section title and hatnote
    # closest to each table (by scanning backward through siblings/ancestors).
    eop_references = soup.find_all('ol', class_="references")   # end-of-page references section (if any)
    eop_references = [refs.find_all('li') for refs in eop_references]
    all_references = {}
    for refs in eop_references:
        for ref in refs:
            all_references[ref.get("id", "")] = extract_text(ref, extract_links=True) # [text, [urls]]
    print(f"Found {len(all_references)} end-of-page references on the page.")

    tables = soup.find_all('table')
    print(f"Found {len(tables)} table(s) on the page.")

    for idx, table in enumerate(tables, 1):
        table_id = f"{page.pageid}-{idx}"
        if args.action == "fix_new_evidence_wikipages":
            if table_ids and table_id in table_ids:
                print(f"  [Action] Fixing new evidence Wikipedia pages for table {table_id} ...")
                data_df, reference_df, metadata = process_table(table)
                wiki_ref_cols = find_wikipedia_reference_columns(reference_df)
                if not wiki_ref_cols:
                    print(f"  [WARN] No Wikipedia reference columns found in table {table_id}.")
                    continue
                ref_col_name, ref_col_idx = wiki_ref_cols[0]
                ref_titles = extract_wiki_titles_from_ref_column(reference_df, ref_col_name)
                metadata["new_evidence_wikipages"] = ref_titles
                output_records.append(dict(
                    table_id=table_id,
                    new_evidence_wikipages=ref_titles,
                    new_evidence_wikipage_urls=reference_df[ref_col_name].tolist(),
                ))
            continue

        print(f"\n--- Table {table_id} ---")
        table_classes = table.get("class", [])
        print("  Table classes: ", table_classes)  # print table classes for debugging

        # -------------------------------------------------------
        # Process special tables in the page (e.g., infobox, navbox) differently if needed
        # -------------------------------------------------------
        # use different parser for infobox
        if any('infobox' in cls for cls in table_classes):
            print("  Detected an infobox table. Process to get extra metadata but skip expansion for now.")
            try:
                data_json = parse_infobox_table(table, verbose=args.debug)
                print(f"  Table content: \n{json.dumps(data_json, indent=2, ensure_ascii=False)}\n")
            except Exception as e:
                print(f"  [WARN] Could not process infobox table {idx}: {e}. Skipping.")
            continue
        
        # skip table of ignored classes
        else:
            ignored_classes = ['navbox', 'sidebar']
            if any([any(_ in cls for cls in table_classes) for _ in ignored_classes]):
                print(f"  Detected a table of ignored class. Skipping for now.")
                continue

        # -------------------------------------------------------
        # Extract table data via the existing helper
        # -------------------------------------------------------
        try:
            data_df, reference_df, metadata = process_table(table)
            print(f"  Table shape: {data_df.shape}")
        except Exception as e:
            print(f"  [WARN] Could not process table {idx}: {e}. Skipping.")
            continue

        if data_df.empty:
            print("  Table is empty. Skipping.")
            continue

        # -------------------------------------------------------
        # Extract section title and hatnote nearest to this table
        # -------------------------------------------------------
        section_title = _find_nearest(table, what="section_title")
        hatnote = _find_nearest(table, what="hatnote")

        # Enrich table_metadata with structural context
        metadata["page_title"] = title
        metadata["section_title"] = section_title
        metadata["hatnote"] = hatnote

        # Collect all references mentioned in the table (e.g., in footnotes) 
        # and link them to the end-of-page references if possible
        table_references = [l
            for col in reference_df.columns \
            for l in reference_df[col].dropna().tolist()
        ]
        table_references.extend([
            *[l for _ in metadata["other_info_references"] for l in _[1] if l],
            *[_[1] for _ in metadata["footer_info"] if _[1]],
            *metadata["caption"][1]
            # metadata["hatnote"][1]
        ])

        # print(f"  Raw table references: {table_references}")
        table_references = [str(_) for _ in table_references if _]  # filter out empty/null references
        table_references = sorted(list(set(table_references)))
        references_content = []
        for ref_id in table_references:
            ref_id = ref_id.lstrip("#")  # remove leading '#' if present    
            references_content.append(all_references.get(ref_id, None))
        metadata["references_content"] = [_ for _ in references_content if _]  # filter out None values

        # metadata["other_info_references"] = [_[0] for _ in metadata["other_info_references"]]
        metadata.pop("other_info_references")
        metadata["footer_info"] = [_[0] for _ in metadata["footer_info"]]
        metadata["caption"] = metadata["caption"][0]

        # -------------------------------------------------------
        # Check regularity before attempting expansion
        # -------------------------------------------------------
        print(f"  Table shape: {data_df.shape}")
        print(f"  Table content: \n{data_df.to_markdown(index=False)}\n")
        # print(f"  Reference DataFrame (for link extraction): \n{reference_df.to_markdown(index=False)}\n")
        print(f"  Table metadata: {json.dumps(metadata, ensure_ascii=False)}")
        print(f"  Section title: '{section_title}'")
        print(f"  Hatnote: '{hatnote}'")

        if not check_table_regularity(data_df):
            print("  Table is not regular. Skipping.")
            continue

        # -------------------------------------------------------
        # Routing logic: determine which method to apply for table expansion
        # -------------------------------------------------------
        # Method 1.1 condition:
        #   The table has at least/most min_rows/max_rows rows and min_columns/max_columns columns
        #       to ensure it's not too small and not too large.
        #   There exists at least one column where ALL (or MIN_WIKI_LINK_FRACTION)
        #   entries have an internal Wikipedia article link.
        wiki_ref_cols = find_wikipedia_reference_columns(reference_df)
        print(f"  Found {len(wiki_ref_cols)} qualifying reference columns for Method 1.1: {[c[0] for c in wiki_ref_cols]}")
        method_1_1_ok = 0

        if wiki_ref_cols and check_table_with_ideal_size(data_df):
            # Loop over all qualifying reference columns (if multiple) and try to expand the table
            # random.choice(wiki_ref_cols) if len(wiki_ref_cols) > 1 else wiki_ref_cols[0]
            # Probably we only consider the first qualifying reference column for now
            # Update on Mar 14: There are cases with many qualifying reference columns.
            ref_col_name, ref_col_idx = wiki_ref_cols[0]  # take the first qualifying reference column for now
            print(f"  Routing to Method 1.1 (reference column: '{ref_col_name}').")
            for _ in range(2):
                try:
                    result = method_1_1_expand_table(
                        args=args,
                        data_df=data_df,
                        reference_df=reference_df,
                        ref_col_name=ref_col_name,
                        table_metadata=metadata,
                    )
                    if result:
                        result = dict(
                            table_id=table_id,
                            page_metadata=page_metadata,
                            table_metadata=metadata,
                            original_table_html=data_df.to_html(index=False),
                            **result,
                        )
                        output_records.append(result)
                        method_1_1_ok = 1
                        break
                except Exception as e:
                    print(f"  [ERROR] Exception during Method 1.1 expansion attempt: {e}")
                    continue
                if method_1_1_ok:
                    break  # stop after the first successful expansion

        if method_1_1_ok == 0:
            # Method 1.2, 2.x etc. — not yet implemented
            print("  Not able to apply Method 1.1 successfully "
                  "(no qualifying reference column or table size too big / small or expansion failed). "
                  "Return the table without expansion for now (can be extended later to implement other methods).")
            result = dict(
                table_id=table_id,
                page_metadata=page_metadata,
                table_metadata=metadata,
                original_table_html=data_df.to_html(index=False),
                expanded_table_html="",  # no expansion
                notes="No expansion applied (no qualifying reference column found).",
                new_columns=[],  # no new columns
                new_evidence_wikipages=[]
            )
            output_records.append(result)

    return output_records


def crawl_pages(args, page_titles: list[str]):
    """
    Crawl a list of Wikipedia page titles, extract and expand tables,
    and return all output records.

    Args:
        args:        Parsed CLI arguments.
        page_titles: List of Wikipedia page titles to process.

    Saves intermediate results to JSONL after each page is processed:
        List of result dicts (one per successfully expanded table).
    """
    if args.output_jsonl_file is not None:
        if os.path.exists(args.output_jsonl_file):
            print(f"Loading JSONL file '{args.output_jsonl_file}")
            fout = open(args.output_jsonl_file, "a", encoding="utf-8")
            # take the last non-empty line to get the last processed page title for resuming
            with open(args.output_jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    last_processed_page = None
                else:
                    last_row = lines[-1]
                    last_row = json.loads(last_row)
                    last_processed_page = last_row["page_metadata"]["page_title"]
        else:
            fout = open(args.output_jsonl_file, "w", encoding="utf-8")
            last_processed_page = None
    else:
        print("No jsonl file specified, skip this function")
        return

    n_pages = len(page_titles)
    if last_processed_page:
        start_index = page_titles.index(last_processed_page) + 1
        page_titles = page_titles[start_index:]
    else:
        start_index = 0

    for i, title in tqdm(enumerate(page_titles, start_index+1),
                         total=len(page_titles), desc=f"Processing wikipages..."):
        try:
            print("="*80)
            print(f"Processing page {i}/{n_pages}: {title}")
            output_records = extract_table_and_context(args, title)

            for record in output_records:
                fout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fout.flush()
        except Exception as e:
            print(f"[ERROR] Exception while processing page '{title}': {e}")
            continue

        if args.debug:
            print("[DEBUG] Debug mode: stopping after first page.")
            break
    fout.close()


def filter_for_good_tables(args):
    """
    As the above pipeline produces a lot of table of various size,
    We want to focus on tables that are of ideal sizes (e.g., 2-6 columns, 5-20 rows).
    """
    if args.output_jsonl_file is not None:
        fin = open(args.output_jsonl_file, "r").readlines()
        fout = open(args.output_jsonl_file.replace(".jsonl", "_filtered.jsonl"), "w")
    else:
        print("No jsonl file specified, skip this function")
        return

    for idx, line in tqdm(enumerate(fin, 1), total=len(fin), desc="Filtering tables with ideal size..."):
        item = json.loads(line)
        if item["expanded_table_html"]:
            df_html = item["expanded_table_html"]
            df = pd.read_html(StringIO(df_html))[0]
        else:
            df_html = item["original_table_html"]
            df = pd.read_html(StringIO(df_html))[0]
            if not check_table_with_ideal_size(df):
                continue
        item["table_shape"] = df.shape
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()


def convert_wikitable_jsonl_to_md(args):
    if args.output_jsonl_file is not None:
        fin = open(args.output_jsonl_file, "r").readlines()
        fout = open(args.output_jsonl_file.replace(".jsonl", ".md"), "w")
    else:
        print("No jsonl file specified, skip this function")
        return

    print("Convert first few samples from JSON to MD format for manual checking and inspection ...")
    for idx, line in enumerate(fin, 1):
        item = json.loads(line)
        if not item.get("expanded_table_html", ""):
            continue
        print(f"Processing line {idx}: JSON object with keys {sorted(list(item.keys()))}")
        html2md = lambda html: pd.read_html(StringIO(html))[0].to_markdown(index=False) if html else "(no table)"

        sample = "\n\n".join([
            "="*40,
            f"# Index {idx}",
            "## Original table\n" + html2md(item.get("original_table_html", "")),
            "## Expanded table\n" + html2md(item.get("expanded_table_html", "")),
            "## Notes\n" + item.get("notes", ""),
            f"## Page:    {item['page_metadata']['page_title']}",
            f"## Section: {item['table_metadata'].get('section_title', 'N/A')}",
            f"## Hatnote: {item['table_metadata'].get('hatnote', 'N/A')}",
            f"## Caption: {item['table_metadata'].get('caption', 'N/A')}",
            "="*5,
        ])
        fout.write(sample)
        fout.flush()

    fout.close()


def fix_new_evidence_wikipages(args):
    """
    Due to an bug that I passed sorted(list(summaries.keys())) (which are table entry valued, not wikipages),
    I need to 
    - re-extract the wiki page titles from the reference column and 
    - fix the new_evidence_wikipages field in the output JSONL
    for expanded tables.
    Other tables are not affected by this bug.
    """
    all_records = [json.loads(l) for l in open(args.output_jsonl_file, "r").readlines()]
    def save():
        fout = open(args.output_jsonl_file.replace(".jsonl", "_fixed.jsonl"), "w", encoding="utf-8")
        for record in all_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.close()

    # gather expanded table
    records_with_expanded_table = []
    table_id_to_record = {}
    for record in all_records:
        if record.get("new_columns", []) and not record.get("new_evidence_wikipage_urls", []):
            records_with_expanded_table.append(record)
            table_id_to_record[record["table_id"]] = record
        if args.debug:
            break
    print(f"Found {len(records_with_expanded_table)} records with expanded tables to fix.")

    page_title_to_list_of_table_ids = {} # {r["page_metadata"]["page_title"]: [] for r in records_with_expanded_table}
    for record in records_with_expanded_table:
        page_title = record["table_metadata"]["page_title"]
        table_id = record["table_id"]
        page_title_to_list_of_table_ids.setdefault(page_title, []).append(table_id)

    # re-crawl new evidence wikipages and update corresponding records
    save_after_n_pages = 20
    for _, page_title, table_ids in enumerate(page_title_to_list_of_table_ids.items(), 1):
        # fixed_records = list[table_id, new_evidence_wikipages, new_evidence_wikipage_urls]
        fixed_records = extract_table_and_context(args, page_title, table_ids)
        for fixed_record in fixed_records:
            table_id = fixed_record["table_id"]
            if table_id in table_id_to_record:
                record = table_id_to_record[table_id]
                record["new_evidence_wikipages"] = fixed_record.get("new_evidence_wikipages", [])
                record["new_evidence_wikipage_urls"] = fixed_record.get("new_evidence_wikipage_urls", [])
                print(f"Fixed new evidence wikipages for table {table_id} on page '{page_title}'.")
        if _ % save_after_n_pages == 0:
            save()
            print(f"Saved progress after processing {_} pages.")

    # save the fixed records back to a new JSONL file
    save()
    print(f"Saved fixed records to '{args.output_jsonl_file.replace('.jsonl', '_fixed.jsonl')}'.")


# -------------------------------------------------------
# Argument parsing  (mirrors gen_code_qa_reasonning_trace.py style)
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawl Wikipedia pages, extract tables, and expand them via Method 1.1."
    )

    parser.add_argument(
        "--action",
        type=str,
        default="crawl",
        choices=["crawl", "demo", "filter", "convert_jsonl_to_md", "fix_new_evidence_wikipages"],
        help=(
            "Action to perform. "
            "'crawl' processes the pages specified by --page_titles_file. "
            "'demo' runs on a hardcoded demo page (Venezuela) for quick testing. "
            "'filter' applies a size-based filter to the output JSONL to keep only tables of ideal size. "
            "'convert_jsonl_to_md' converts the output JSONL to a human-readable MD format for inspection. "
            "'fix_new_evidence_wikipages' fix the new_evidence_wikipages field in the output JSONL by re-extracting wiki links."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only the first page and print verbose logs.",
    )

    # ---- Input / output ----
    parser.add_argument(
        "--page_titles_file",
        type=str,
        default="datasets/ours/v2/wikipedia_new_pages.txt",
        help="Path to a plain-text file with one Wikipedia page title per line.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="""We can specify a partition [total_partitions]_[index] of the data to process for each run,
        e.g. '4_1' to process the first quarter of the data, '4_2' to process the second quarter, etc."""
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        default="datasets/ours/v2/wikipedia_new_and_expanded_tables.jsonl",
        help="Path to write the output JSONL file.",
    )

    # ---- LLM parameters (same style as gen_code_qa_reasonning_trace.py) ----
    parser.add_argument(
        "--model",
        type=str,
        default="hosted_vllm/gpt-oss-120b",
        help="LLM model identifier passed to litellm.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level for the LLM.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key or endpoint URL for the LLM (used with hosted_vllm models).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://0.0.0.0:8000/v1",
        help="Base URL for the LLM API (used with hosted_vllm models).",
    )
    parser.add_argument(
        "--api_rpm_limit",
        type=int,
        default=None,
        help="Rate limit in requests-per-minute for the LLM API. "
             "If set, a sleep is inserted between calls.",
    )

    args = parser.parse_args()

    # Derive sleep duration from RPM limit (same pattern as gen_code_qa_reasonning_trace.py)
    if args.api_rpm_limit:
        args.llm_call_sleep = int(60.0 / args.api_rpm_limit)
    else:
        args.llm_call_sleep = 0
    if args.partition:
        # args.output_file = args.output_file.replace('.jsonl', f'_{args.partition}.jsonl')
        total_partitions, index = map(int, args.partition.split('_'))
        assert 1 <= index <= total_partitions, "Partition index must be between 1 and total partitions"

    return args


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    # wikipedias = [
    #   MediaWiki(user_agent="search-and-aggregate (qtq.the.channel@gmail.com)"),
    #   MediaWiki(user_agent="fanoutqa/1.0.0 (andrz@seas.upenn.edu)"),
    #   MediaWiki()
    # ]
    wikipedia = MediaWiki()
    random.seed(2026)
    
    if args.action == "demo":
        # Quick smoke-test on a known page with tables
        demo_titles = ["Venezuela", "2026 Winter Olympics", "COVID-19 vaccination in Taiwan",
                       "Tornadoes of 2026", "Cristiano Ronaldo"]
        print(f"[Demo] Running on: {demo_titles}")
        crawl_pages(args, demo_titles)
    
    elif args.action == "crawl":
        # Collect page titles from file
        page_titles = open(args.page_titles_file, 'r', encoding='utf-8').read().splitlines()
        # page_titles = random.sample(page_titles, min(500, len(page_titles)))  # sample a subset for testing
        if args.partition:
            total_partitions, index = map(int, args.partition.split('_'))
            start_index = (index-1)*len(page_titles)//total_partitions
            end_index = index*len(page_titles)//total_partitions
            page_titles = page_titles[start_index:end_index]
            print(f"Processing partition {index}/{total_partitions} of the data with {len(page_titles)} items")
        print(f"[Crawl] Processing {len(page_titles)} page title(s).")
        crawl_pages(args, page_titles)
        filter_for_good_tables(args)

    elif args.action == "filter":
        filter_for_good_tables(args)

    elif args.action == "convert_jsonl_to_md":
        convert_wikitable_jsonl_to_md(args)

    elif args.action == "fix_new_evidence_wikipages":
        fix_new_evidence_wikipages(args)

