"""
Merge document generation and QAR generation JSONL files,
then post-process the data into training format (input, thinking, output).
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


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

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON(L) file and handle null/empty lines."""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                item = json.loads(line)
                if item:  # Skip null objects
                    data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {file_path}")


def remove_attribute_tags(text: str) -> str:
    """Remove attribute tags like <team_home>, <team_away>, etc."""
    # Remove opening and closing tags
    text = re.sub(r'<[^>]+>', '', text)
    return text


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
    artifacts = [
        "Purpose and action in natural language:",
        "**Purpose and action in natural language:**",
        "Corresponding code snippet:",
        "**Corresponding code snippet:**",
        "The corresponding code snippet to support the logic:",
    ]
    intermediate_result_phrases = [ 
        "Resulting intermediate variable:",
        "Resulting intermediate variables:",
        "**The resulting intermediate variable:**",
    ]
    
    for artifact in artifacts:
        reasoning_traces = reasoning_traces.replace(artifact, "")
    for phrase in intermediate_result_phrases:
        reasoning_traces = reasoning_traces.replace(phrase, "The intermediate result is:\n")

    # Clean up excessive whitespace and newlines
    reasoning_traces = re.sub(r'\n{3,}', '\n\n', reasoning_traces)
    reasoning_traces = reasoning_traces.strip()
    
    return reasoning_traces


def replace_variables_with_values(reasoning_traces: str, python_code: str, df_markdown: str) -> str:
    """
    Execute Python code to get variable values, then replace placeholders
    in reasoning traces with actual values.
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
        df = load_table_from_markdown(df_markdown)
        
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


def load_table_from_markdown(md_table: str) -> pd.DataFrame:
    """Load a pandas DataFrame from markdown table string."""
    # Remove extra whitespace and split into lines
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    
    # Find the header and separator lines
    if len(lines) < 2:
        raise ValueError("Invalid markdown table format")
    
    # Skip the index column if present (first column with |---|)
    # Parse header
    header_line = lines[0]
    headers = [col.strip() for col in header_line.split('|') if col.strip()]
    
    # Skip separator line (lines[1])
    
    # Parse data rows
    data_rows = []
    for line in lines[2:]:
        cells = [col.strip() for col in line.split('|') if col.strip()]
        if cells:
            data_rows.append(cells)
    
    # Create DataFrame
    try:
        df = pd.DataFrame(data_rows, columns=headers)
    except:
        try:
            df = pd.DataFrame(data_rows, columns=['Index'] + headers)
        except Exception as e:
            print(f"Error creating DataFrame from markdown table: {e}")
            raise ValueError("Failed to parse markdown table into DataFrame")
    
    # Try to infer and convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails
    
    return df


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def merge_data(doc_gen_file: str, qar_gen_file: str) -> List[Dict[str, Any]]:
    """
    Merge document generation and QAR generation JSONL files.
    
    Returns:
        List of merged dictionaries with all relevant fields.
    """
    print(f"Loading document generation file: {doc_gen_file}")
    doc_data = load_jsonl(doc_gen_file)
    
    print(f"Loading QAR generation file: {qar_gen_file}")
    qar_data = load_jsonl(qar_gen_file)
    
    # Create mapping: table_id -> generated_doc_set
    print("Creating table_id to document mapping...")
    table_id_mapping = {}
    for item in doc_data:
        table_id = item.get('table')
        generated_doc_set = item.get('Generated Document Set', None)
        if generated_doc_set:
            table_id_mapping[table_id] = {
                'generated_doc_set': generated_doc_set,
                'df': item.get('Resultant Generated Table Markdown', None),
            }
    
    print(f"Found {len(table_id_mapping)} document sets")
    
    # Merge with QAR data
    print("Merging data...")
    merged_data = []
    for item in qar_data:
        table_id = item.get('table_id')
        if table_id not in table_id_mapping:
            print(f"Warning: No document set found for table_id: {table_id}. Skipping this sample.")
            continue
        
        merged_item = {
            'table_id': table_id,
            'table_meta': item.get('metadata', []),
            'generated_doc_set': table_id_mapping[table_id]['generated_doc_set'],
            'df': table_id_mapping[table_id]['df'],
            'question': item.get('question'),
            'python_code': item.get('python_code'),
            'final_answer': item.get('final_answer_sql'),
            'reasoning_traces': item.get('reasoning_traces'),
            'reasoning_traces_full': item.get('reasoning_traces_full'),
        }
        merged_data.append(merged_item)
    
    print(f"Successfully merged {len(merged_data)} items")
    return merged_data


def postprocess_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Post-process a single merged item into (input, thinking, output) format.
    
    Returns:
        Dictionary with 'input', 'thinking', 'output' keys, or None if processing fails.
    """
    try:
        # Check all fields
        for key in ['generated_doc_set', 'question', 'df',
                    'python_code', 'final_answer', 'reasoning_traces_full']:
            if not item.get(key, ''):
                print(f"Warning: Missing {key} for table_id: {item.get('table_id')}")
                return None

        # Extract fields
        doc_set = item.get('generated_doc_set')
        question = item.get('question')
        df_markdown = item.get('df')
        python_code = item.get('python_code')
        final_answer = item.get('final_answer')
        reasoning_traces_full = item.get('reasoning_traces_full')
        
        # ---- Process INPUT ----
        # Combine documents (removing attribute tags)       
        if isinstance(doc_set, list):
            documents_text = "\n\n".join([
                f"**Document {i+1}**:\n{remove_attribute_tags(re.sub('[\n ]+', ' ', doc))}"
                for i, doc in enumerate(doc_set)
            ])
        else:
            documents_text = remove_attribute_tags(str(doc_set))
        
        input_text = (
            NARRATIONS["input_prefix"] +
            documents_text +
            NARRATIONS["input_question_prefix"] +
            question +
            NARRATIONS["input_question_suffix"]
        )
        
        # ---- Process THINKING ----
        # Clean and enhance reasoning traces
        reasoning_clean = clean_reasoning_traces(reasoning_traces_full)
        
        # Replace variable placeholders with actual values
        reasoning_with_values = replace_variables_with_values(
            reasoning_clean,
            python_code,
            df_markdown
        )
        
        thinking_text = (
            NARRATIONS["thinking_table_intro"] +
            df_markdown +
            NARRATIONS["thinking_reasoning_intro"] +
            reasoning_with_values
        )
        
        # ---- Process OUTPUT ----
        output_text = NARRATIONS["output_prefix"] + str(final_answer)
        
        return {
            'table_id': item.get('table_id'),
            'input': input_text,
            'thinking': thinking_text,
            'output': output_text,
            'metadata': {
                'question': question,
                'num_documents': len(doc_set) if isinstance(doc_set, list) else 1,
                'table_meta': item.get('table_meta', []),
            }
        }
        
    except Exception as e:
        print(f"Error processing item {item.get('table_id')}: {e}")
        return None


def main(args):
    """Main processing pipeline."""
    print("="*60)
    print("Starting merge and post-processing pipeline")
    print("="*60)
    
    # Step 1: Merge data
    merged_data = merge_data(args.doc_gen_file, args.qar_gen_file)
    if args.debug:
        merged_data = merged_data[:10]  # Limit to first 10 items for debugging
    n_samples = len(merged_data)
    
    # Step 2: Post-process each item
    print("\nPost-processing items...")
    processed_data = []
    for i, item in enumerate(merged_data, 1):
        print(f"Processing item {i}/{n_samples}: {item.get('table_id')}")
        processed_item = postprocess_item(item)
        if processed_item:
            processed_data.append(processed_item)
    
    print(f"\nSuccessfully processed {len(processed_data)}/{n_samples} items")
    
    # Step 3: Save processed data
    if processed_data:
        output_file = args.output_file or args.qar_gen_file.replace('.jsonl', '_processed.jsonl')
        save_jsonl(processed_data, output_file)
        
        # Save a sample as markdown for inspection
        sample_md_file = output_file.replace('.jsonl', '_sample.md')
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
        description="Merge and post-process document generation and QAR generation data"
    )
    parser.add_argument(
        "--doc_gen_file",
        type=str,
        required=True,
        help="Path to document generation JSON(L) file"
    )
    parser.add_argument(
        "--qar_gen_file",
        type=str,
        required=True,
        help="Path to QAR generation JSON(L) file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: qar_gen_file with _processed suffix)"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug mode with verbose logging"
    )
    
    args = parser.parse_args()
    main(args)


