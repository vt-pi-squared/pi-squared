import json
from bs4 import BeautifulSoup
from helper import extract_text


def parse_nested_table(table):
    """Parse a nested table recursively."""
    rows = table.find_all('tr', recursive=False)
    result = []
    
    for row in rows:
        cells = row.find_all(['th', 'td'], recursive=False)
        if len(cells) == 1:
            result.append(extract_text(cells[0]))
        elif len(cells) == 2:
            result.append({
                extract_text(cells[0]): extract_text(cells[1])
            })
        else:
            result.append([extract_text(cell) for cell in cells])
    
    return result


def parse_infobox_row(row):
    """Parse a single infobox row into appropriate structure."""
    # Check for single-column rows
    full_data = row.find(class_=lambda x: x and ('infobox-full-data' in x or 'infobox-below' in x))
    if full_data:
        # Check for nested tables
        nested_table = full_data.find('table')
        if nested_table:
            return parse_nested_table(nested_table)
        return extract_text(full_data)
    
    # Get label and data cells
    label_cell = row.find(class_=lambda x: x and 'infobox-label' in x)
    data_cells = row.find_all(class_=lambda x: x and 'infobox-data' in x)
    
    if not label_cell:
        # Row without label (might be a continuation or special row)
        if len(data_cells) == 1:
            nested_table = data_cells[0].find('table')
            if nested_table:
                return parse_nested_table(nested_table)
            return extract_text(data_cells[0])
        elif len(data_cells) > 1:
            return [extract_text(cell) for cell in data_cells]
        return None
    
    label = extract_text(label_cell)
    
    if not data_cells:
        # Label without data
        return {label: ""}
    
    if len(data_cells) == 1:
        # Check for nested table in data cell
        nested_table = data_cells[0].find('table')
        if nested_table:
            return {label: parse_nested_table(nested_table)}
        return {label: extract_text(data_cells[0])}
    
    # Multiple data cells - return as list
    return {label: [extract_text(cell) for cell in data_cells]}


def extract_footnotes(soup):
    """Extract footnotes and other metadata from the table."""
    footnotes = []
    
    # Look for footnote sections (typically at the bottom)
    footnote_section = soup.find(class_=lambda x: x and 'ib-country-fn' in x)
    if footnote_section:
        notes = footnote_section.find_all('li')
        for note in notes:
            footnotes.append(extract_text(note))
    
    return footnotes


def parse_infobox_table(table, verbose=False):
    """
    Parse an infobox table from bs4 into JSON format.
    
    Args:
        table: a bs4 object representing the table
    
    Returns:
        result: a Python dict representing the parsed table, or None if not an infobox
    """    
    if 'bs4' not in str(type(table)):
        raise ValueError("Expected a bs4 BeautifulSoup object for the table element.")

    # Check if it's an infobox
    table_classes = table.get('class', [])
    if not any('infobox' in cls for cls in table_classes):
        return None
    
    result = {
        "type": "infobox",
        "sections": {},
        "other_info": {}
    }
    
    # Extract title/caption
    caption = table.find('caption')
    if caption:
        result["caption"] = extract_text(caption)
    
    # Get all rows
    rows = table.find_all('tr')
    
    current_section = None
    current_section_name = None
    n_rows = len(rows)
    
    for _, row in enumerate(rows, 1):
        # Check if this row is a section header
        header = row.find(class_=lambda x: x and 'infobox-header' in x)
        
        if header or (_ == n_rows):
            # Save previous section if exists
            if current_section is not None and current_section:
                if current_section_name in result["sections"]:
                    result["sections"][current_section_name].extend(current_section)
                else:
                    result["sections"][current_section_name] = current_section
            
            # Start new section
            if header:
                current_section_name = extract_text(header)
                if current_section_name == "":
                    current_section_name = "General"
                current_section = []
                if verbose:
                    print(f"Found section header: {current_section_name}")
        else:
            # Parse regular row
            parsed_row = parse_infobox_row(row)
            if parsed_row is not None:
                if current_section is None:
                    # No section header yet, create default section
                    current_section = []
                    current_section_name = "General"
                current_section.append(parsed_row)
    
    # if all elements in a section are dictionaries, convert it to a single dictionary
    for section, content in result["sections"].items():
        if all(isinstance(item, dict) for item in content):
            merged_content = {}
            for item in content:
                merged_content.update(item)
            content = merged_content
            result["sections"][section] = content

    # Extract footnotes and other info
    # footnotes = extract_footnotes(soup)
    # if footnotes:
    #     result["other_info"]["footnotes"] = footnotes
    
    # Extract image info if present
    image = table.find(class_='infobox-image')
    if image:
        img_tag = image.find('img')
        if img_tag:
            result["other_info"]["image_url"] = img_tag.get('src', '')
        caption_div = image.find(class_='infobox-caption')
        if caption_div:
            result["other_info"]["image_caption"] = extract_text(caption_div)
    
    return result


def parse_html_file(file_path):
    """Parse all infobox tables from an HTML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator
    tables_html = content.split('#-----#')
    
    results = {}
    for i, table_html in enumerate(tables_html):
        table_html = table_html.strip()
        if not table_html:
            continue
        
        # Find and parse the infobox
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        parsed = parse_infobox_table(table_html, verbose=True) if table else None
        if parsed:
            results[i] = parsed
    
    return results


if __name__ == "__main__":
    # Example usage
    file_path = "datasets/ours/v2/wikipedia_infobox_samples.html"
    results = parse_html_file(file_path)
    
    # Optionally save to file
    with open("datasets/ours/v2/wikipedia_infobox_parsed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nParsed {len(results)} infobox tables successfully!")


