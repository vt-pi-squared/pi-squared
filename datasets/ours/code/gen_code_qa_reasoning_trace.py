import os, re, json
import pandas as pd
from io import StringIO
from pathlib import Path
FILE_DIR = Path(__file__)

from wikipedia_table_crawler import check_table_with_ideal_size
from dspy_program_qar_gen import *
from helper import *

from sqlite3 import connect
conn = connect(":memory:")



def load_table_from_markdown_demo():
    md_table = """
    |    | club         | sport      |   founded | league                | venue               |
    |---:|:-------------|:-----------|----------:|:----------------------|:--------------------|
    |  0 | fk vojvodina | football   |      1914 | jelen superliga       | kara\u0111or\u0111e stadium   |
    |  1 | fk proleter  | football   |      1951 | first league          | stadion slana bara  |
    |  2 | fk novi sad  | football   |      1921 | first league          | detelinara stadium  |
    |  3 | kk vojvodina | basketball |      2000 | sinalco superleague   | spens sports center |
    |  4 | kk novi sad  | basketball |      1985 | sinalco superleague   | spens sports center |
    |  5 | ok vojvodina | volleyball |      1946 | serbian volley league | spens sports center |
    |  6 | hk vojvodina | hockey     |      1957 | serbian hockey league | spens sports center |
    |  7 | hk novi sad  | hockey     |      1998 | serbian hockey league | spens sports center |
    """

    df = load_table_from_markdown(md_table)
    print(df.dtypes)
    df.to_sql(name="df", con=conn, index=False, if_exists='replace')
    # print(pd.read_sql("SELECT * FROM df", conn))


def convert_qar_jsonl_to_md(args):
    if args.output_jsonl_file is not None:
        fin = open(args.output_jsonl_file, "r").readlines()
        fout = open(args.output_jsonl_file.replace(".jsonl", ".md"), "w")
    else:
        print("No jsonl file specified, skip this function")
        return

    print("Convert first < 50 samples from JSON to MD format for manual checking and inspection ...")
    for idx, line in enumerate(fin[:50]):
        item = json.loads(line)
        print(f"Processing line {idx+1}: JSON object with keys {item.keys()}")
        if item.get('reasoning_traces', None) is None:
            continue

        sample = "\n\n".join([
            "="*40,
            f"# Index {idx+1}: {item['table_id']}",
            f"## Metadata:\n```python\n{item['metadata']}\n```",
            f"## Table:\n{item['df']}",
            f"## New Columns (as expanded in the collection process):\n{item.get('new_columns', [])}",
            f"## Condition Column:\n{item.get('condition_col', 'N/A')}",
            f"## Question:\n{item['question']}",
            f"## Question Quality:\n```python\n{item.get('question_quality', 'N/A')}\n```",
            f"## Generated SQL Query:\n```sql\n{item.get('sql_query', 'N/A')}\n```",
            f"## Generated Python Code:\n```python\n{item.get('python_code', 'N/A')}\n```",
            f"## Final Answer (SQL):\n{item.get('final_answer_sql', 'N/A')}",
            f"## Final Answer (Python):\n{item.get('final_answer_python', 'N/A')}",
            f"## SQL vs Python Comparison Judgment:\n{item.get('dfc_judgment', 'N/A')}",
            f"## Reasoning Traces:\n{item.get('reasoning_traces', 'N/A')}",
            f"## Reasoning Traces Full:\n{item.get('reasoning_traces_full', 'N/A')}",
            "="*5,
        ])
        fout.write(sample)
        fout.flush()

    fout.close()


def search_for_avoid_keyword(question):
    avoided_keywords = [
        'list', 'table', 'column', 'row', 'field',
        'section', 'caption', 'metadata', 
        'dataframe', 'markdown', 'html', 'json', 'sql', 'python',
        'this',
    ]
    question = question.lower()
    for word in avoided_keywords:
        if word in question:
            return word
    return None


# Generate query (likely with answer and reasoning traces) for each table, using LLM
def generate_query(args, tables):
    """
    Generate queries (and answers) for each table using LLM, and save the results in a JSONL file.
    
    Input:
        - args: the arguments containing the settings for generation
        - tables: a list of tables, where each table is a dictionary with keys
            "table_id": str,
            "metadata": JSON object,
            "df" : pd.DataFrame
            "condition_col": str (optional, the column to used for adding conditions in question generation)
            "new_columns": list of str (optional, the new columns added in the expansion process, which can be used in question generation)
            "notes": str (optional, the notes from the expansion process, which can be used in question generation)

    Each line in the JSONL file is a JSON object with the following keys:
        - table_id: the id of the table
        - metadata: the metadata of the table (page title, section title, caption, etc.)
        - df: the markdown/string format of the table
        - question: the generated question which is complex, self-contained, and natural
        - question_quality: the quality of the generated question regarding complexity, self-containedness, and naturalness
        - sql_query: the generated SQL query
        - python_code: the generated Python code
        - final_answer_sql: the final answer obtained by executing the SQL query
        - final_answer_python: the final answer obtained by executing the Python code
        - dfc_judgment: the judgment on whether the SQL answer and Python answer are the same (or close enough)
        - reasoning_traces: the reasoning traces translated from the Python code (if SQL and Python answers are the same)
    """
    if args.output_jsonl_file is not None:
        if os.path.exists(args.output_jsonl_file) and not args.debug:
            print(f"Loading JSONL file '{args.output_jsonl_file}'")
            fout = open(args.output_jsonl_file, "a", encoding="utf-8")
            # take the last non-empty line to get the last processed page title for resuming
            with open(args.output_jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    last_processed_table = None
                else:
                    last_row = lines[-1]
                    last_row = json.loads(last_row)
                    last_processed_table = last_row["table_id"]
                    print(f"Last processed table_id: {last_processed_table}")
        else:
            fout = open(args.output_jsonl_file, "w", encoding="utf-8")
            last_processed_table = None
    else:
        print("No jsonl file specified, skip this function")
        return
    
    table_ids = [table['table_id'] for table in tables]
    n_pages = len(table_ids)
    if last_processed_table:
        start_index = table_ids.index(last_processed_table) + 1
        tables = tables[start_index:]
    else:
        start_index = 0

    for _, table in enumerate(tables, start_index+1):
        table_id = table['table_id']
        print("="*20 + f"\nTable {_}/{n_pages} w/ id {table_id}:")
        
        try:
            metadata = table['metadata']
            df = table['df']
            print(metadata, df, df.dtypes, sep="\n")
            print("Simplifying table to markdown/string format ...")
            df_mdstr = simplify_to_md_or_str(df)
            metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)

            print("Add table to in-memory SQLite database ...")
            conn.execute("DROP TABLE IF EXISTS df")
            df.to_sql(name="df", con=conn, index=False, if_exists='replace')

            print("Preparing messages for LLM ...")
            messages = get_messages_templates(SQaGenerator, demos=SQaGenerator_demos)
            messages[-1]["content"] = messages[-1]["content"].format(
                table=df_mdstr,
                metadata=metadata_str,
                condition_col=table.get("condition_col", '(No condition column)'),
                notes=table.get("notes", ""),
            )
            assert df_mdstr in messages[-1]["content"], "Not found the table!"

        except Exception as e:
            print(f"Error processing {table_id}: {e}")
            print("Skip this table ...")
            continue


        for _ in range(1, args.max_trials+1):
            response = {}
            try:
                # generate qa and sql
                response = llm_call(args=args, messages=messages)
                response = SQaGenerator_parser(response)
                print(json.dumps(response, indent=2))
                response["table_id"] = table_id
                response["condition_col"] = table.get("condition_col", '(No condition column)')

                # simply check if the question has some avoided keywords that refer to table or wikipedia items
                word = search_for_avoid_keyword(response["question"])
                if word:
                    raise ValueError(f"The generated question '{response['question']}' contains avoided keyword(s) '{word}'.")

                # check if the question's quality (qq) is good and get the revision if can be improved
                qq_messages = get_messages_templates(QuestionQualityChecker, demos=QuestionQualityChecker_demos)
                qq_messages[-1]["content"] = qq_messages[-1]["content"].format(
                    question=response["question"],
                    # metadata=metadata_str
                )
                qq_response = llm_call(args=args, messages=qq_messages)
                qq_response = QuestionQualityChecker_parser(qq_response)
                print("Question quality check response:\n", qq_response)
                response["question_quality"] = qq_response
                # filter by rating (scale from 1 to 5)
                if qq_response['complexity_rating'] < 3:
                    raise ValueError("The generated question is not complex enough, retrying ...")
                if qq_response['self_containedness_rating'] < 4:
                    raise ValueError("The generated question is not self-contained, retrying ...")
                if qq_response['naturalness_rating'] < 4:
                    raise ValueError("The generated question is not natural enough, retrying ...")
                if qq_response.get('revision', ''):
                    print("The question can be improved. Revised question:\n", qq_response['revision'])
                    # response["question"] = qq_response['revision']

                # retrieve and execute sql_query
                sql_query = response["sql_query"]
                # if sql_query has ```sql ...``` , extract the inner part
                while True:
                    match = re.search(r"```sql(.*?)```", sql_query, re.DOTALL)
                    if match:
                        sql_query = match.group(1).strip()
                    else:
                        break
                response["sql_query"] = sql_query # save SQL code w/o Markdown formatting

                print('Executing SQL query:\n"""', sql_query, '"""\n')
                final_answer_sql = pd.read_sql(sql_query, conn)

                for _python in range(1, args.max_trials+1):
                    try:
                        # generate python code
                        py_messages = get_messages_templates(PyGenerator)
                        py_messages[-1]["content"] = py_messages[-1]["content"].format(
                            table=df_mdstr,
                            metadata=metadata_str,
                            question=response["question"],
                        )
                        python_gen_response = llm_call(args=args, messages=py_messages)
                        python_gen_response = PyGenerator_parser(python_gen_response)

                        # retrieve and execute python_code
                        python_code = python_gen_response["python_code"]
                        # if python_code has ```python ...``` , extract the inner part
                        match = re.search(r"```python(.*?)```", python_code, re.DOTALL)
                        if match:
                            python_code = match.group(1).strip()
                        response["python_code"] = python_code  # save the independently generated python code

                        print('Executing Python code:\n"""', python_code, '"""\n')
                        # output should be stored in variable `final_answer_python`
                        df = table['df']
                        local_vars = locals()
                        exec(python_code, globals(), local_vars)
                        final_answer_python = local_vars['final_answer_python']

                        # compare final_answer_sql and final_answer_python
                        print("Final answer from SQL query:", final_answer_sql)
                        print("Final answer from Python code:", final_answer_python)

                        # TODO: have some heuristic to compare final_answer_sql and final_answer_python
                        # use LLM to check if final_answer_sql and final_answer_python are the same (or close enough)
                        dfc_messages = get_messages_templates(DfComparison)
                        dfc_messages[-1]["content"] = dfc_messages[-1]["content"].format(
                            question=response["question"],
                            metadata=metadata_str,
                            result1=simplify_to_md_or_str(final_answer_sql),
                            result2=simplify_to_md_or_str(final_answer_python)
                        )
                        comparison_response = llm_call(args=args, messages=dfc_messages)
                        comparison_response = DfComparison_parser(comparison_response)

                        response["metadata"] = metadata
                        response["df"] = df_mdstr
                        response["dfc_judgment"] = comparison_response["judgment"]
                        response["final_answer_sql"] = simplify_to_md_or_str(final_answer_sql)
                        response["final_answer_python"] = simplify_to_md_or_str(final_answer_python)

                        # if the results are the same, generate reasoning traces from Python codes
                        print("Comparison result between SQL and Python answers:", comparison_response)
                        if comparison_response["judgment"]:
                            print("The results produced by the SQL query and Python code are the same")
                            print("Generating reasoning traces from the Python code...")

                            rt_messages = get_messages_templates(CodeInterpretation)
                            rt_messages[-1]["content"] = rt_messages[-1]["content"].format(
                                metadata=metadata_str,
                                question=response["question"],
                                python_code=python_code
                            )
                            rt_response = llm_call(args=args, messages=rt_messages)
                            rt_result = CodeInterpretation_parser(rt_response)
                            print("Reasoning traces translated from the Python code:\n", rt_result["interpretation"])
                            response["reasoning_traces"] = rt_result["interpretation"]
                            try:
                                local_vars = {k: simplify_to_md_or_str(v) for k, v in local_vars.items()}
                                response["reasoning_traces_full"] = rt_result["interpretation"].format(**local_vars)
                            except:
                                response["reasoning_traces_full"] = rt_result["interpretation"]
                            fout.write(json.dumps(response) + "\n")
                            fout.flush()
                            break

                        else:
                            raise ValueError("The results produced by the SQL query and Python code are different")

                    except Exception as e:
                        print("Error in Python code generation/execution or result comparison:", e)
                        if _python < args.max_trials:
                            print("="*5 + f"\nRetrying {_python+1}-th time for Python code generation ...")
                            continue
                        else:
                            print("Max trials for Python code generation reached, save `response` with SQL part only and move on ...")
                            if response:
                                fout.write(json.dumps(response) + "\n")
                                fout.flush()
                                break

                # If the execution can reach this line, then the generated question is good. 
                # Can break the trial loop
                break

            except Exception as e:
                print("Error occurred:", e)
                if _ < args.max_trials:
                    print("="*5 + f"\nRetrying {_+1}-th time ...")
                    continue
                else:
                    print("Max trials reached, save `response` anyway and move on ...")
                    if response:
                        fout.write(json.dumps(response) + "\n")
                        fout.flush()

        if args.debug:
            break

    fout.close()


def filter_at_most_two_question_per_table(args):
    """
    Filter for successfully generated questions with reasoning traces, and keep at most 2 questions per table to ensure diversity.
    The current code already only generates one question per table,
    """
    if args.output_jsonl_file is not None:
        fin = open(args.output_jsonl_file, "r").read().splitlines()
        fout = open(args.output_jsonl_file.replace(".jsonl", "_filtered.jsonl"), "w")
    else:
        print("No jsonl file specified, skip this function")
        return

    seen_table_ids = {}
    for line in fin:
        item = json.loads(line)
        try:
            table_id = item["table_id"]
        except Exception as e:
            print(f"Error processing line: {e}")
            print(f"Skipping line because of missing 'table_id': {line}\n{item}")
            continue
        if seen_table_ids.get(table_id, 0) < 2 and item.get('reasoning_traces', None):
            fout.write(json.dumps(item) + "\n")
            seen_table_ids.setdefault(table_id, 0)
            seen_table_ids[table_id] += 1

    fout.close()
    print(f"Filtering done. Output saved to {args.output_jsonl_file.replace('.jsonl', '_filtered.jsonl')}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--action",
        type=str,
        default="generate_query",
        choices=["generate_query", "load_table_demo", "filter", "convert_jsonl_to_md"],
        help="Action to perform."
    )
    parser.add_argument("--debug",
        action="store_true",
        help="Whether to run in debug mode (only process one table and print more logs)."
    )

    # Data generation settings
    parser.add_argument("--max_trials",
        type=int,
        default=3,
        help="Maximum number of attempts to generate a valid query that can be executed without errors and produces a non-empty answer."
    )
    parser.add_argument("--table_source",
        type=str,
        default="ours_10",
        choices=["ours_all", "ours_10", "demo"],
        help="Source of the table to generate the query on."
    )
    parser.add_argument("--input_tab_doc_jsonl_file",
        type=str,
        default="datasets/ours/v2/tab_doc_expansion.jsonl",
        help="The input JSONL file containing the table documents (used when table_source is 'ours')."
    )
    parser.add_argument("--output_jsonl_file",
        type=str,
        default=None,
        help="File to save the generated queries and results."
    )

    # LLM generation parameters
    parser.add_argument("--model",
        type=str,
        default="hosted_vllm/gpt-oss-120b",
        help="LLM model to use for query generation."
    )
    parser.add_argument("--reasoning_effort",
        type=str,
        default="low",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level for the LLM."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key or endpoint URL for the LLM (used with hosted_vllm models).",
    )
    parser.add_argument("--base_url",
        type=str,
        default="http://0.0.0.0:8000/v1",
        help="Base URL for the LLM API, if applicable."
    )
    parser.add_argument("--api_rpm_limit",
        type=int,
        default=None,
        help="Rate limit (requests per minute) for the LLM API, if applicable."
    )

    args = parser.parse_args()
    if args.api_rpm_limit:
        args.llm_call_sleep = int(60.0 / args.api_rpm_limit) + 1
    else:
        args.llm_call_sleep = 0
    return args



if __name__ == "__main__":
    args = parse_args()

    if args.action == "load_table_demo":
        load_table_from_markdown_demo()

    elif args.action == "generate_query":
        if args.table_source == "demo":
            # load demo table
            load_table_from_markdown_demo()
            df = pd.read_sql("SELECT * FROM df", conn)
            metadata = {
                "page_title": "Sports Clubs in Novi Sad",
                "page_id": "",
                "section_title": "Summary of Sports Clubs",
                "caption": "Sports Clubs in Novi Sad"
            }
            tables = [{"table_id": "na", "metadata": metadata, "df": df}]

        elif "ours" in args.table_source:
            # JSONL. crawled from Wikipedia, saved table in HTML format.
            tab_doc = [json.loads(l) for l in open(args.input_tab_doc_jsonl_file, "r").readlines()]
            if "10" in args.table_source:
                tab_doc = tab_doc[:50]  # only use the first 10 tables for quick debugging and testing
            tables = []
            for item in tab_doc:
                if item["expanded_table_html"]:
                    df_html = item["expanded_table_html"]
                    df = pd.read_html(StringIO(df_html))[0]
                else:
                    df_html = item["original_table_html"]
                    df = pd.read_html(StringIO(df_html))[0]
                    if not check_table_with_ideal_size(df):
                        print(f"Skip table {item["table_id"]} due to too large / small size")
                        continue

                new_columns = item.get("new_columns", [])
                condition_col = new_columns[0] if len(new_columns) > 0 else '(No condition column)'
                metadata = item["table_metadata"]
                # metadata["page_summary"] = item["page_metadata"]["page_summary"]
                tables.append({"table_id": item["table_id"], "metadata": metadata, "df": df, 
                               "new_columns": new_columns, "condition_col": condition_col, "notes": item["notes"]})

        generate_query(args, tables)
        filter_at_most_two_question_per_table(args)

    elif args.action == "filter":
        filter_at_most_two_question_per_table(args)

    elif args.action == "convert_jsonl_to_md":
        convert_qar_jsonl_to_md(args)
