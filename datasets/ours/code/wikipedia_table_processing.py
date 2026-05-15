import re, json, copy
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Any
from helper import extract_text, df_drop_empty_row_column_and_convert_types



def get_table_metadata(table_element) -> Dict:
    """
    Process an bs4 table element and extract metadata.
    """
    
    # Extract table caption if exists
    caption = ["", []]
    caption_element = table_element.find('caption')
    if caption_element:
        caption = extract_text(caption_element, extract_links=True)
    
    # Extract all rows
    all_rows = table_element.find_all('tr')
    
    # Extract other info and references from headers rows
    other_rows = [row for row in all_rows if row.find('th')]
    other_info_references = []
    for row in other_rows:
        cells = row.find_all('th')
        for cell in cells:
            cell_text = extract_text(cell)
            
            # Extract reference URLs from links
            references = []
            for link in cell.find_all('a', href=True):
                href = link['href']
                references.append(href)
            other_info_references.append([cell_text, references])
    
    # Create metadata dictionary
    metadata = {
        "caption": caption,
        "other_info_references": other_info_references
    }
    
    return metadata


def resolve_repeated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect if columns are perfectly repeated (with .1, .2 suffixes) and if so,
    split the DataFrame horizontally and stack vertically.
    Supports both flat and MultiIndex column names.
    """
    # print(f"Original columns: {df.columns}")
    is_multiindex = isinstance(df.columns, pd.MultiIndex)

    def strip_suffix(col):
        if is_multiindex:
            # Only strip suffix from the last level
            return col[:-1] + (re.sub(r'\.\d+$', '', col[-1]),)
        return re.sub(r'\.\d+$', '', col)

    base_names = [strip_suffix(c) for c in df.columns]

    # Find unique base names while preserving order
    seen = {}
    for col, base in zip(df.columns, base_names):
        seen.setdefault(base, []).append(col)

    # Check if there is a repeated pattern (i.e., any base name appears more than once)
    max_repeat = max(len(v) for v in seen.values())
    if max_repeat < 2:
        return df  # No repetition, nothing to do

    # Reconstruct repeated column groups
    unique_bases = list(seen.keys())
    n_groups = max_repeat

    chunks = []
    for group_idx in range(n_groups):
        group_cols = []
        for base in unique_bases:
            candidates = seen[base]
            if group_idx < len(candidates):
                group_cols.append(candidates[group_idx])
            else:
                # This group is missing this column — skip whole group
                group_cols = None
                break
        if group_cols is None:
            continue

        chunk = df[group_cols].copy()
        if is_multiindex:
            chunk.columns = pd.MultiIndex.from_tuples(unique_bases)
        else:
            chunk.columns = list(unique_bases)
        chunks.append(chunk)

    if not chunks:
        return df

    result = pd.concat(chunks, ignore_index=True)
    return result


def process_table(table_element):
    """
    Process an bs4 table element and extract data, references, and metadata.
    """
    # Get metadata
    metadata = get_table_metadata(table_element)
    metadata['header_levels'] = []
    metadata['footer_info'] = []

    # Read the table with links and save the first column as index
    data_df = pd.read_html(StringIO(str(table_element)), extract_links='body')[0]
    if isinstance(data_df, pd.Series):
        data_df = data_df.to_frame()

    # Rename index column if it's unnamed. But fine, LLMs can understand its meaning
    # for i in range(1,6):
    #     if f'Unnamed: {i}' in data_df.columns:
    #         data_df.rename(columns={'Unnamed: {i}': 'Info{i}'}, inplace=True)

    # Remove header level if it only has one value
    while isinstance(data_df.columns, pd.MultiIndex):
        highest_level = [_ for _ in data_df.columns.levels[0] if not re.match(r'Unnamed: \d+', str(_))]
        if len(highest_level) <= 1:
            metadata['header_levels'].append(highest_level[0])
            data_df.columns = data_df.columns.droplevel(0)
        else:
            break

    # NOTE: the following command converts multi-level column name to single-level column names
    # to simplify the processing with Markdown tables
    if isinstance(data_df.columns, pd.MultiIndex):
        fn = lambda cols: ' -- '.join(
            [c for c in cols[:-1] if not re.match(r'Unnamed: \d+', str(c))] + [cols[-1]])
        data_df.columns = data_df.columns.map(fn)

    # Clean `[]` from column names
    data_df = df_drop_empty_row_column_and_convert_types(data_df)
    is_multiindex = isinstance(data_df.columns, pd.MultiIndex)
    new_columns = []
    for col in data_df.columns:
        if isinstance(col, tuple):
            new_col = tuple(re.sub(r'\s*\[[^\]]*\]', '', str(c)).strip() for c in col)
        else:
            new_col = re.sub(r'\s*\[[^\]]*\]', '', str(col)).strip()
        new_columns.append(new_col)
    data_df.columns = pd.MultiIndex.from_tuples(new_columns) if is_multiindex else new_columns

    # Move the last row to metadata if it has <= 2 unique values while the n_cols is larger
    if len(data_df) > 1:
        last_row = data_df.iloc[-1]
        last_row = sorted([_ for _ in list(set(last_row)) if _[0]]) # _ still has links
        if (len(last_row) <= 2) and (len(data_df.columns) > 2):
            metadata['footer_info'] = last_row
            data_df = data_df.iloc[:-1]

    # Resolve the repeated columns by unfolding them (if it's the case)
    data_df = resolve_repeated_columns(data_df)

    # Split reference_df from data_df
    data_w_links_df = copy.deepcopy(data_df)
    data_df = data_w_links_df.map(lambda v: v[0])
    reference_df = data_w_links_df.map(lambda v: v[1])

    # Clean the `[ ]` citation note in data_df
    def clean_cell(v):
        if isinstance(v, str):
            # replace '\n' by ';' for better Markdown table processing
            v = re.sub('\n', '; ', v).strip()
            new_v = re.sub(r'\s*\[[^\]]*\]', '', v).strip()
            return new_v if new_v else v
        return v
    for col in data_df.columns:
        data_df[col] = data_df[col].apply(clean_cell)

    return data_df, reference_df, metadata


def main(file_path):
    """Parse all tables from an HTML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    tables_html = content.split('#-----#')

    results = {}
    for i, table_html in enumerate(tables_html, 1):
        print(f"\n{'='*40}\n")
        print(f"Processing table {i}/{len(tables_html)}...")

        table_html = table_html.strip()
        if not table_html:
            return None
        else:
            # Find and parse the table
            soup  = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
            if table is None:
                return None
        
        data_df, reference_df, metadata = process_table(table)
        if data_df is not None and not data_df.empty:
            print(
                f"Extracted DataFrame with shape {data_df.shape}\n\n{data_df}\n\n" + \
                f"Extracted Reference:\n\n{reference_df}\n\n" + \
                f"Extracted Metadata:\n\n{metadata}\n"
            )
        parsed = {
            "table": data_df.to_html() if data_df is not None else None,
            "references": reference_df.to_html() if reference_df is not None else None,
            "metadata": metadata
        }
        if parsed:
            results[i] = parsed

    return results


if __name__ == "__main__":
    # Example usage
    file_path = "datasets/ours/v2/wikipedia_table_samples.html"
    results = main(file_path)

    # Optionally save to file
    with open("datasets/ours/v2/wikipedia_table_parsed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nParsed {len(results)} tables successfully!")
