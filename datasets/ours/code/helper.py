import os
import re
import json
import time
import litellm
import pandas as pd
from utils import count_tokens


def llm_call(args, messages, get_thinking_tokens=False):
    kwargs = {}
    if "qwen3.5" in args.model.lower():
        kwargs = dict(
            temperature=0.7, top_p=0.8, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
    if args.model.startswith("hosted_vllm/"):
        kwargs["base_url"] = args.base_url
        if args.api_key:
            kwargs["api_key"] = args.api_key

    # if count_tokens(messages[-1]["content"]) > int(1024*128*0.9):
    #     print("Warning: the input to LLM is too long, use more than 90 percent of context length. Skip it.")
    #     return None

    for _ in range(1, 4):
        try:
            time.sleep(args.llm_call_sleep)
            response = litellm.completion(
                timeout=100,
                model=args.model,
                messages=messages,
                reasoning_effort=args.reasoning_effort,
                # max_completion_tokens=1024
                # base_url=args.base_url,
                **kwargs
            )
            if not get_thinking_tokens:
                return response["choices"][0]["message"]["content"]
            else:
                return response["choices"][0]["message"]["content"], response["choices"][0]["message"].get("reasoning_content", "")
        except Exception as e:
            print("LLM call error:", e)
            print(f"LLM call re-trying {_}-th time ...")
            time.sleep(2)


def simplify_to_md_or_str(df):
    if type(df) in [pd.DataFrame, pd.Series]:
        # check if size of df is (1,1)
        shape = df.shape
        if shape == (1, 1):
            return str(df.iloc[0, 0])
        elif shape[1] == 1: # only one column, convert to list
            return str(df.iloc[:, 0].tolist())
        elif shape[0] == 1: # only one row, convert to dict
            return str(df.iloc[0, :].to_dict())
        else:
            try:
                return df.to_markdown(index=False)
            except:
                for col in df.columns:
                    df[col] = df[col].astype(str)
                return df.to_markdown(index=False)
    else:
        return str(df)

"""
import mdpd
def load_table_from_markdown(markdown_str: str):
    df = mdpd.from_md(markdown_str)
    df.fillna('', inplace=True)
    # print(df.dtypes, df.info())

    # drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)

    # remove index columns if exists
    df.drop(columns=['Unnamed: 0', ''], inplace=True, errors='ignore') 

    # convert columns to appropriate/best data types
    df = df.convert_dtypes()
    for col in df.columns:
        try:
            # convert to numeric
            df[col] = pd.to_numeric(df[col])
        except:
            try:
                # convert to list/dict if possible
                # df[col] = df[col].apply(eval)
                # convert to datetime
                # df[col] = pd.to_datetime(df[col])
                pass
            except:
                pass
    return df
"""


def load_table_from_markdown(md_table: str) -> pd.DataFrame:
    """Load a pandas DataFrame from markdown table string."""
    # Remove extra whitespace and split into lines
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    
    # Find the header and separator lines
    if len(lines) < 2:
        raise ValueError("Invalid markdown table format")
    
    # Parse header
    header_line = lines[0]
    headers = [col.strip() for col in header_line.split('|')]
    while headers and headers[-1] == '':
        headers.pop()
    
    # Skip separator line (lines[1])
    
    # Parse data rows
    data_rows = []
    for line in lines[2:]:
        cells = [col.strip() for col in line.split('|')]
        while cells and cells[-1] == '':
            cells.pop()
        if cells:
            data_rows.append(cells)
    
    # Create DataFrame
    try:
        if headers[0].lower() == '':
            headers[0] = 'Index'
        df = pd.DataFrame(data_rows, columns=headers)
    except:
        try:
            df = pd.DataFrame(data_rows, columns=['Index'] + headers)
        except Exception as e:
            print(f"Error creating DataFrame from markdown table: {e}")
            raise ValueError("Failed to parse markdown table into DataFrame")
    
    # If the `Index`` column is exactly index from 0 to n-1, then drop it and reset index
    if 'Index' in df.columns:
        try:
            df['Index'] = pd.to_numeric(df['Index'])
            if df['Index'].tolist() == list(range(len(df))):
                df.drop(columns=['Index'], inplace=True)
        except:
            pass

    return df_drop_empty_row_column_and_convert_types(df)


def df_drop_empty_row_column_and_convert_types(df):
    # drop rows and columns with all missing values
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # convert columns to appropriate/best data types
    df = df.convert_dtypes()
    
    # Try to infer and convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails

    return df


def extract_text(element, extract_links=False):
    """Extract clean text from a BeautifulSoup HTML element, preserving some structure."""
    if element is None:
        return ""
    
    references = []
    for link in element.find_all('a', href=True):
        href = link['href']
        references.append(str(href))
    
    # Replace <br> with newlines to preserve line breaks in text
    for br in element.find_all('br'):
        br.replace_with('\n')
    
    # Get text and clean up whitespace
    text = element.get_text(separator=' ', strip=False)
    # Remove `[ ]` footnote markers only if they are not the entire text
    # (to avoid removing valid content or the case of reference column)
    footnote_free_text = re.sub(r'\[[\d ]+\]', '', text).strip()
    text = footnote_free_text if footnote_free_text else text
    text = text.replace('[ edit ]', '')  # remove "[ edit ]" in Wikipedia section headers
    # Normalize whitespace
    text = re.sub(r'[ ]+', ' ', text).strip()
    
    return [text, references] if extract_links else text


def prepare_and_load_cache(output_file, all_samples, identifier_key):
    last_row = {}
    if output_file is not None:
        if os.path.exists(output_file):
            print(f"Loading JSONL file '{output_file}'")
            fout = open(output_file, "a", encoding="utf-8")
            # take the last non-empty line to get the last processed page title for resuming
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) != 0:
                    last_row = json.loads(lines[-1])
        else:
            fout = open(output_file, "w", encoding="utf-8")
    else:
        print("No jsonl file specified, skip this function")
        return
    
    all_ids = [sample[identifier_key] for sample in all_samples]
    if last_row and identifier_key in last_row:
        id_ = last_row[identifier_key]
        start_index = all_ids.index(id_) + 1
    else:
        start_index = 0
    
    return fout, start_index
