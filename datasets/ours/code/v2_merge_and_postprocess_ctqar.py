import os
import re
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from io import StringIO
from typing import Dict, List, Any, Optional

from wikipedia_table_crawler import crawl_wikipedia_page, WIKI_CACHE_DIR
from gather_relevant_web_articles_for_qa import WEB_CACHE_DIR
from utils import bm25plus_context_compaction, count_tokens
from markdownify import MarkdownConverter


# ============================================================================
# NARRATIONS - Easy to edit later
# ============================================================================

NARRATIONS = {
    "input_prefix": "Given the following documents:\n\n",
    "input_question_prefix": "\n\nQuestion: ",
    "input_question_suffix": "\n(Please answer within the scope of the given documents.)",
    
    "thinking_table_intro": "To answer this question, I first aggregate the key information from the documents into a structured table:\n\n",
    "thinking_reasoning_intro": "\n\nNow I'll work through the solution step by step:\n\n",
    
    "output_prefix": "Based on the analysis, the answer is:\n\n",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {file_path}")


# Markdownification
def yeet(*_):
    return ""

class MDConverter(MarkdownConverter):
    def convert_img(self, el, text, convert_as_inline):
        alt = el.attrs.get("alt", None) or ""
        return f"![{alt}](image)"
    
    def convert_a(self, el, text, convert_as_inline):
        return text
    
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def convert_div(self, el, text, convert_as_inline):
        content = text.strip()
        if not content:
            return ""
        return f"{content}\n"
    
    # sometimes these appear inline and are just annoying
    convert_script = yeet
    convert_style = yeet


def get_markdownified_wikipedia_page(args, title: str) -> str:
    """
    Get a Wikipedia page and convert HTML to Markdown, with custom handling for images and links.
    NOTE: Sections, paragraphs or tables are well separated with triple new line char '\n\n\n'
    Allow caching.
    """
    cache_file_path = os.path.join(WIKI_CACHE_DIR, f"{title.replace('/', '-').replace('\\', '-')}.md")
    if os.path.exists(cache_file_path):
        md = open(cache_file_path, "r", encoding='utf-8').read()
        if args.debug:
            print(f"Loaded markdownified Wikipedia page {title} from cache {cache_file_path}")
        return md
    else:
        md = MDConverter(heading_style="atx").convert(crawl_wikipedia_page(title, args.debug).html)
        with open(cache_file_path, "w", encoding='utf-8') as f:
            f.write(md)
        if args.debug:
            print(f"Saved markdownified Wikipedia page {title} to cache {cache_file_path}")
        return md


# Refine reasoning traces
def clean_reasoning_traces(reasoning_traces: str) -> str:
    """
    Clean reasoning traces by:
    1. Removing Python code blocks
    2. Removing repetitive keywords/artifacts
    3. Making it more human-like
    """
    if not reasoning_traces:
        return ""
    
    # Remove Python code blocks wrapped in ```python ... ```
    reasoning_traces = re.sub(r'```python.*?```', '', reasoning_traces, flags=re.DOTALL)
    
    # Remove repetitive keywords/artifacts
    # Hardcode bthe behavior of Qwen3.5-35B-A3B
    artifacts = [
        "Purpose and action in natural language",
        "Corresponding code snippet"
    ]
    intermediate_result_phrases = [ 
        "Intermediate variable",
    ]
    
    lines = reasoning_traces.splitlines()
    cleaned_lines = []
    for line in lines:
        if any(artifact.lower() in line.lower() for artifact in artifacts):
            continue  # Skip lines containing artifacts
        if any(phrase.lower() in line.lower() for phrase in intermediate_result_phrases):
            cleaned_lines.append("The intermediate result is:")  # Standardize intermediate result phrasing
        else:
            cleaned_lines.append(line)
    
    # Clean up excessive whitespace and newlines
    reasoning_traces = re.sub(r'\n{3,}', '\n\n', "\n".join(cleaned_lines)).strip()
    return reasoning_traces


def replace_variables_with_values(reasoning_traces: str, python_code: str, df_html: str) -> str:
    """
    Execute Python code to get variable values,
    then replace placeholders in reasoning traces with actual values.
    """
    if not reasoning_traces or not python_code:
        return reasoning_traces
    
    # Find all variable placeholders in format {variable_name}
    placeholders = re.findall(r'\{(\w+)\}', reasoning_traces)
    if not placeholders:
        return reasoning_traces
    
    # Prepare execution environment
    try:
        # Load the dataframe
        df = pd.read_html(StringIO(df_html))[0]
        
        # Execute the Python code
        local_vars = {'df': df}
        print(f"Executing Python code to replace variables: {placeholders}")
        exec(python_code, globals(), local_vars)
        
        # Replace placeholders with actual values
        for var_name in placeholders:
            if var_name in local_vars:
                value = local_vars[var_name]
                # Convert value to string representation
                if isinstance(value, pd.DataFrame):
                    value_str = value.to_markdown(index=False)
                elif isinstance(value, (list, dict)):
                    value_str = json.dumps(value, indent=2, ensure_ascii=False)
                else:
                    value_str = str(value)
                
                # Replace placeholder
                reasoning_traces = reasoning_traces.replace(f'{{{var_name}}}', value_str)
        
    except Exception as e:
        print(f"Warning: Could not execute Python code to replace variables: {e}")
        # Return cleaned traces even if execution fails
    
    return reasoning_traces



# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def merge_data_fn(args) -> List[Dict[str, Any]]:
    # Load data
    print(f"Loading Table and metadata file: {args.tab_doc_file}")
    tab_doc_data = [json.loads(l) for l in open(args.tab_doc_file, "r").readlines()]

    print(f"Loading QAR file: {args.qar_file}")
    qar_data = [json.loads(l) for l in open(args.qar_file, "r").readlines()]

    print(f"Loading web articles file: {args.websearch_file}")
    websearch_data = [json.loads(l) for l in open(args.websearch_file, "r").readlines()]
    
    # Create 2 mapping:
    #   question -> web articles
    #   table_id -> original wiki page + new_evidence_wikipages
    question_to_web = {item["question"]: item for item in websearch_data}
    table_id_to_wiki = {}
    for item in tab_doc_data:
        df_html = item["expanded_table_html"] if item["expanded_table_html"] else item["original_table_html"]
        evidence_wikipages = [item["table_metadata"]["page_title"]] + item["new_evidence_wikipages"]
        table_id_to_wiki[item["table_id"]] = {"df_html": df_html, "evidence_wikipages": evidence_wikipages}

    # Merge data
    print("Merging data ...")
    merged_data = []
    for item in qar_data:
        if item.get('reasoning_traces', None):
            table_id = item['table_id']
            question = item['question']
            for k in ['final_answer_python', 'dfc_judgment', 'reasoning_traces']:
                item.pop(k, None)
            if table_id not in table_id_to_wiki:
                print(f"Warning: table_id {table_id} not found in tab_doc_data, skipping")
                continue
            if question not in question_to_web:
                print(f"Warning: question '{question}' not found in websearch_data, skipping web data")
                continue
            if question_to_web[question]["table_id"] != table_id:
                print(f"Warning: question '{question}' in websearch_data does not match table_id {table_id}, skipping web data")
                continue

            item['reasoning_traces'] = item.pop('reasoning_traces_full', "")
            item['df_md'] = item.pop('df', "")
            item.update(table_id_to_wiki[table_id])
            item['relevant_web_urls'] = question_to_web[question]['relevant_web_urls']
            item['relevant_web_articles'] = question_to_web[question]['relevant_web_articles']
            merged_data.append(item)

    print(f"Successfully merged {len(merged_data)} items")
    return merged_data


def split_sentences_if_needed(chunk: str, max_tokens: int=1024) -> List[str]:
    """
    Chunk markdown text into pieces with at most max_tokens tokens using `.`.
    Split then accumulate sentences until reaching the max_tokens limit, then start a new chunk.
    """
    sentences = re.split(r'(?<=[.?!])\s+', chunk.strip())
    # re.split(r'(?<=[.?!])\s+', "Hi! How are you? I'm fine.") -> ["Hi!", "How are you?", "I'm fine."]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(re.split(r'\s+', sentence)) > max_tokens:
            # If a single sentence exceeds max_tokens, we have to use it anyway
            chunks.append(current_chunk)
            chunks.append(sentence)
            current_chunk = ""
            continue
        
        if len(re.split(r'\s+', current_chunk)) + len(re.split(r'\s+', sentence)) <= max_tokens:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_markdown_avoid_block_or_table(md: str, max_tokens: int=1024) -> List[str]:
    """
    Chunk markdown text into pieces with at most max_tokens tokens, while avoiding breaking code blocks or tables.
    This is a helper function for context compaction.
    """
    # For simplicity, we can just split by double newlines and then group them into chunks
    if len(re.split(r'\s+', md.strip())) <= max_tokens:
        return [md.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    cursor_position = "text" # ["text", "code", "table"]
    
    for section in md.split("\n"):
        section = section.strip()
        if not section:
            continue
        section_tokens = len(re.split(r'\s+', section))
        
        if not current_chunk:
            current_chunk = section
            current_tokens = section_tokens
        elif cursor_position == "text" and not section.startswith("|") and not section.startswith("```"):
            if current_tokens + section_tokens > max_tokens:
                chunks.extend(split_sentences_if_needed(current_chunk.strip()))
                current_chunk = section
                current_tokens = section_tokens
            else:
                current_chunk += "\n" + section
                current_tokens += section_tokens
        elif cursor_position == "text" and section.startswith("|"):
            chunks.extend(split_sentences_if_needed(current_chunk.strip()))
            current_chunk = section
            current_tokens = section_tokens
            cursor_position = "table"
        elif cursor_position == "table" and section.startswith("|"):
            current_chunk += "\n" + section
            current_tokens += section_tokens
        elif cursor_position == "table" and not section.startswith("|"):
            chunks.append(current_chunk.strip())
            current_chunk = section
            current_tokens = section_tokens
            cursor_position = "text"
        elif cursor_position == "text" and section.startswith("```"):
            chunks.extend(split_sentences_if_needed(current_chunk.strip()))
            current_chunk = section
            current_tokens = section_tokens
            cursor_position = "code"
        elif cursor_position == "code" and section.startswith("```"):
            current_chunk += "\n" + section
            current_tokens += section_tokens
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_tokens = 0
            cursor_position = "text"
        elif cursor_position == "code":
            current_chunk += "\n" + section
            current_tokens += section_tokens
    
    if current_chunk:
        chunks.extend(split_sentences_if_needed(current_chunk.strip()))
    
    return [chunk for chunk in chunks if chunk]


def get_section_content_by_title(md: str, section_title: str) -> Optional[str]:
    """
    Extract the content of a section in markdown text by its title.
    """
    pattern = rf'(#+\s*{re.escape(section_title)}\s*)'
    parts = re.split(pattern, md, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 3:
        return None  # Section not found
    section_content = parts[2].split("\n#", 1)[0]  # Get content until the next section
    return parts[1] + section_content


def postprocess_item(args, item: Dict[str, Any], item_order: int) -> Optional[Dict[str, Any]]:
    """
    Post-process a single merged item
    - Get the wikipages and web articles ready, compact them with BM25+
    - (Optional) Get reasoning traces again if usual substitution failed
    """
    # Get the wikipages and web articles ready
    print(f"Loading and markdownifying wikipages and web articles for question: {item['question']}")
    print(f"Wikpages: {item['evidence_wikipages']}")
    anchor_page = get_markdownified_wikipedia_page(args, item["metadata"]["page_title"])
    wikipages = [get_markdownified_wikipedia_page(args, title) for title in item['evidence_wikipages'][1:]]
    web_articles = [open(os.path.join(WEB_CACHE_DIR, f), 'r').read() for f in item['relevant_web_articles']]

    # Compact context
    # Set to int(1024*128*0.75), # 25% of context is preserved for output, like Claude Code
    # but should set it randomly to encourage more diversity
    # Observed context length in benchmarks + evaluation settings --> mean 54k, std 30k, max 96k ~ 75%, min 2k
    # preserve extra 3k tokens for each pages
    random.seed(item_order)
    n_tokens = int(1024*(random.normalvariate(mu=54, sigma=30) + len(wikipages + web_articles)*3))
    n_tokens = max(min(n_tokens, 1024*96), 1024*4)  # Clamp between 4k and 96k tokens

    # prechunked = [[_.strip() for _ in w.split("\n\n")] for w in ([anchor_page] + wikipages + web_articles)]
    prechunked = []
    prechunked_must_keep = []   # ensure the infoboxes of evidence wikipages are in the context
    for w in wikipages:
        chunks = []
        # get the first 10k char of the wikipage
        # as it covers the first infobox - a sufficient information of the question 
        infobox = w[:10000].rsplit("\n\n")[0]
        n_tokens -= count_tokens(infobox)
        prechunked_must_keep.append(infobox)
        w = w[len(infobox):]  # remove the infobox part to avoid duplication
        for piece in w.split("\n\n"):
            chunks.extend(chunk_markdown_avoid_block_or_table(piece, 256))
        prechunked.append(chunks)
    for w in web_articles:
        chunks = []
        for piece in w.split("\n\n"):
            chunks.extend(chunk_markdown_avoid_block_or_table(piece, 256))
        prechunked.append(chunks)

    # process the main wikipage (in markdown format) that contains the table that the question is developed from
    section_title = item["metadata"]["section_title"].rsplit(" / ")[-1]
    section_containing_table = get_section_content_by_title(anchor_page, section_title)
    if section_containing_table:
        anchor_parts = anchor_page.split(section_containing_table, 1)
    else:
        section_containing_table = anchor_page
        anchor_parts = ["", ""]
    n_tokens -= count_tokens(section_containing_table)
    for w in anchor_parts:
        chunks = []
        for piece in w.split("\n\n"):
            chunks.extend(chunk_markdown_avoid_block_or_table(piece, 256))
        prechunked.append(chunks)

    print(f"Compacting context with BM25+ for question: {item['question']}")
    print(f"Total number of chunks before compaction: {sum(len(chunks) for chunks in prechunked)}")
    if n_tokens > 0:
        context = bm25plus_context_compaction(
            max_remaining_tokens=n_tokens,
            documents=prechunked,
            main_instruction=item['question'],
        )
        if isinstance(context, list) and isinstance(context[0], list):
            context = ["\n".join(chunks) for chunks in context]
    else:
        context = ["" for _ in prechunked]

    # merge the prechunked_must_keep back
    for _, must_keep in enumerate(prechunked_must_keep):
        context[_] = must_keep + '\n\n' + context[_]
    anchor_page_chunked = context[-2] + '\n' + section_containing_table + '\n' + context[-1]
    context = [anchor_page_chunked] + context[:-2]

    random.shuffle(context)
    if isinstance(context, list) and len(context) > 1:
        context = "\n\n".join(["="*10 + f"\n**Document {i}**: \n\n\n{''.join(doc)}" \
                                for i, doc in enumerate(context, 1)])
    else:
        context = context[0]

    # Replace markdown heading level L ('#' x L) in pages and articles with equivalent section/subsection keywords.
    # to avoid confusion with markdown headings in reasoning traces or prompt
    cleaned_lines = []
    for line in context.splitlines():
        # find max '#' string at the beginning of the line
        match = re.match(r'^(#+)', line.strip())
        if match:
            heading_marker = match.group(1)
            heading_level = len(heading_marker)
            if heading_level == 1:
                line = line.replace(heading_marker, '**Document Title**: ')
            elif heading_level >= 2:
                section_word = 'sub'*(heading_level-2) + 'section'
                section_word = 'S' + section_word[1:] # e.g. Subsubsection
                line = line.replace(heading_marker, f'*{section_word}*: ')
        cleaned_lines.append(line)
    context = "\n".join(cleaned_lines)

    if args.form_naive_reasoning_trace:
        # ----------
        # Forming naive end-to-end training data with input, thinking, and output
        # ----------     
        # INPUT   
        input_text = (
            NARRATIONS["input_prefix"] +
            context +
            NARRATIONS["input_question_prefix"] +
            item.get('question') +
            NARRATIONS["input_question_suffix"]
        )
        # THINKING
        try:
            print(f"Cleaning and enhancing reasoning traces for question: {item['question']}")
            # Clean and enhance reasoning traces
            reasoning_clean = clean_reasoning_traces(item['reasoning_traces'])
            # Replace variable placeholders with actual values
            item['reasoning_traces'] = replace_variables_with_values(
                reasoning_clean, item.get('python_code'), item.get('df_html')
            )
        except:
            pass
        thinking_text = (
            NARRATIONS["thinking_table_intro"] +
            item.get('df_md') +
            NARRATIONS["thinking_reasoning_intro"] +
            item['reasoning_traces']
        )
        # OUTPUT
        output_text = NARRATIONS["output_prefix"] + str(item.get('final_answer'))
        item.update({'input': input_text, 'thinking': thinking_text, 'output': output_text})
    else: # just pack the context and let LLM generate the full reasoning trace from context to answer
        item['context'] = context
    return item


def main(args):
    """Main processing pipeline."""
    print("="*60)
    print("Starting merge and post-processing pipeline")
    print("="*60)
    
    # Step 1: Merge data
    print("Merging data from different sources...")
    if args.merged_file:
        if os.path.exists(args.merged_file):
            print(f"Loading merged data from {args.merged_file}")
            merged_data = [json.loads(l) for l in open(args.merged_file, "r").readlines()]
        else:
            print(f"Merged file {args.merged_file} not found, merging data from scratch")
            merged_data = merge_data_fn(args)
            save_jsonl(merged_data, args.merged_file)
    else:
        merged_data = merge_data_fn(args)
    
    if args.partition:
        total_partitions, index = map(int, args.partition.split('_'))
        start_index = (index-1)*len(merged_data)//total_partitions
        end_index = index*len(merged_data)//total_partitions
        merged_data = merged_data[start_index:end_index]
        print(f"Processing partition {index}/{total_partitions} of the data with {len(merged_data)} items")
    
    # Step 2: Post-process each item
    print("\nPost-processing items...")
    last_row = {}
    if args.output_file is not None:
        if os.path.exists(args.output_file):
            print(f"Loading JSONL file '{args.output_file}'")
            fout = open(args.output_file, "a", encoding="utf-8")
            # take the last non-empty line to get the last processed page title for resuming
            with open(args.output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) != 0:
                    last_row = json.loads(lines[-1])
        else:
            fout = open(args.output_file, "w", encoding="utf-8")
    else:
        print("No jsonl file specified, skip this function")
        return
    
    # use the question string as question_id
    n_samples = len(merged_data)
    question_ids = [item['question'] for item in merged_data]
    if last_row and "question" in last_row:
        if last_row["question"] not in question_ids:
            print(f"Warning: last processed question '{last_row['question']}' not found in merged data, "
                  "likely due to the merge data is truncated. Stop processing.")
            return
        start_index = question_ids.index(last_row["question"]) + 1
        merged_data = merged_data[start_index:]
    else:
        start_index = 0
    
    if args.debug:
        merged_data = merged_data[:10]  # Limit to first 10 items for debugging
    processed_data = []
    for i, item in tqdm(enumerate(merged_data, start_index+1), total=len(merged_data), desc="Post-processing items"):
        try:
            print(item.get('table_id')) # f"=====\nProcessing item {i}/{n_samples}: {}
            processed_item = postprocess_item(args, item, i)
            print(f"Finished processing item {i}/{n_samples}: {item.keys()}\n")
            if processed_item:
                processed_data.append(processed_item)
                fout.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                print(f"Warning: post-processing failed for item {item.get('table_id')}, skipping")
        except:
            print(f"Error processing item {item.get('table_id')}, skipping")
            continue
    print(f"\nSuccessfully processed {len(processed_data)}/{n_samples} items")
    fout.close()
    
    # Step 3: Save a sample as markdown for inspection
    if processed_data:
        if args.form_naive_reasoning_trace:
            sample_md_file = args.output_file.replace('.jsonl', '_sample.md')
            with open(sample_md_file, 'w', encoding='utf-8') as f:
                for _, sample in enumerate(processed_data[:10], 1):
                    sample_str = '\n\n'.join([
                        f"# Table {_}, ID {sample['table_id']}",
                        f"## Metadata\n\n```python\n{sample['metadata']}\n```",
                        f"## INPUT\n\n{sample['input']}",
                        f"## THINKING\n\n{sample['thinking']}",
                        f"## OUTPUT\n\n{sample['output']}",
                        "="*5,
                        ""
                    ])
                    f.write(sample_str)
                    f.flush()
                f.close()
            print(f"Saved sample to {sample_md_file}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge and post-process collected table + metadata, generated QAR, and web articles of Q"
    )
    parser.add_argument(
        "--tab_doc_file",
        type=str,
        required=True,
        help="Path to collected table and other metadata JSONL file"
    )
    parser.add_argument(
        "--qar_file",
        type=str,
        required=True,
        help="Path to generated QAR JSONL file"
    )
    parser.add_argument(
        "--websearch_file",
        type=str,
        default=None,
        help="Path to websearch result JSONL file"
    )
    parser.add_argument(
        "--merged_file",
        type=str,
        default=None,
        help="Path to the merged JSONL file. Act as as an intermediate file " \
        "that contains the merged data before post-processing. If not specified, it will not be saved."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: qar_file with _processed suffix)"
    )
    parser.add_argument(
        "--form_naive_reasoning_trace",
        action='store_true',
        help="Whether to form the generated Python-guided context-jump reasoning trace for training"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="""Because compacting the context take quite some time,
        we can specify a partition [total_partitions]_[index] of the data to process for each run,
        e.g. '4_1' to process the first quarter of the data, '4_2' to process the second quarter, etc."""
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug mode with verbose logging"
    )

    args = parser.parse_args()
    if not args.websearch_file:
        args.websearch_file = args.qar_file.replace(".jsonl", "_websearch.jsonl")
    if not args.output_file:
        args.output_file = args.qar_file.replace('.jsonl', '_processed.jsonl')
    if args.partition:
        # args.output_file = args.output_file.replace('.jsonl', f'_{args.partition}.jsonl')
        total_partitions, index = map(int, args.partition.split('_'))
        assert 1 <= index <= total_partitions, "Partition index must be between 1 and total partitions"

    main(args)
