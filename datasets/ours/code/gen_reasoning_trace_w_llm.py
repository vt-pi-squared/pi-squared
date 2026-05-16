"""
Reasoning Trace Generation Pipeline

Generates high-quality reasoning trajectories for question-answering tasks using the Back Translation technique 
(using the known correct answer to guide the generation of answer-agnostic reasoning steps).
"""

import json
from tqdm import tqdm
from helper import llm_call, prepare_and_load_cache


def generate_reasoning_traces(args):
    prompt_template = open(args.prompt_template_file, "r").read()
    all_samples = [json.loads(l) for l in open(args.input_file, 'r', encoding='utf-8').readlines() if l.strip()]
    n_samples = len(all_samples)
    identifier_key = 'question'
    
    fout, start_index = prepare_and_load_cache(args.output_file, all_samples, identifier_key)
    all_samples = all_samples[start_index:]
    
    for _, sample in tqdm(enumerate(all_samples, start_index+1), total=len(all_samples)):
        sample.pop('llm_generated_reasoning_traces', '')
        try:
            print("="*20 + f"\nSample {_}/{n_samples}: {sample[identifier_key]}")
            messages = [{"role": "user", "content": prompt_template.format(
                context=sample["context"],
                question=sample["question"],
                # table=sample["df_md"],
                answer=sample["final_answer_sql"]
            )}]
            response, thinking_content = llm_call(args, messages, get_thinking_tokens=True)
            if response:
                sample['llm_generated_reasoning_traces'] = response
                sample['llm_thinking_content'] = thinking_content
                fout.write(json.dumps(sample) + "\n")
                fout.flush()
            else:
                print(f"\tError generating reasoning trace for sample with {identifier_key} '{sample[identifier_key]}'")
            if args.debug:
                print(f"\t[DEBUG] Generated reasoning trace:\n{response}")
                print(f"\t[DEBUG] Finish processing the first sample, exiting due to debug mode.")
                break
        except Exception as e:
            pass
    fout.close()


def convert_jsonl_to_md(args):
    if args.output_file is not None:
        fin = open(args.output_file, "r").readlines()
        fout = open(args.output_file.replace(".jsonl", ".md"), "w")
    else:
        print("No jsonl file specified, skip this function")
        return

    print("Convert first < 50 samples from JSON to MD format for manual checking and inspection ...")
    for idx, line in enumerate(fin[:10]):
        item = json.loads(line)
        print(f"Processing line {idx+1}: JSON object with keys {item.keys()}")
        if not item.get('llm_generated_reasoning_traces', None):
            continue

        sample = "\n\n".join([
            "="*40,
            f"# Index {idx+1}: {item['table_id']}",
            f"## Metadata:\n```python\n{item['metadata']}\n```",
            f"## Context:\n```python\n{item['context']}\n```",
            f"## Table:\n{item['df_md']}",
            f"## Condition Columm:\n{item['condition_col']}",
            f"## Question:\n{item['question']}",
            f"## Correct Answer:\n{item.get('final_answer_sql', 'N/A')}",
            f"## Question Quality:\n```python\n{item.get('question_quality', 'N/A')}\n```",
            f"## Generated SQL Query:\n```sql\n{item.get('sql_query', 'N/A')}\n```",
            f"## LLM Generated Reasoning Trace:\n```text\n{item['llm_generated_reasoning_traces']}\n```",
            "="*5,
        ])
        fout.write(sample)
        fout.flush()

    fout.close()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--action",
        type=str,
        default="generate_reasoning_traces",
        choices=["generate_reasoning_traces", "filter", "convert_jsonl_to_md"],
        help="Action to perform."
    )
    parser.add_argument("--debug",
        action="store_true",
        help="Whether to run in debug mode (only process one table and print more logs)."
    )

    # Reasoning trace generation settings
    parser.add_argument("--prompt_template_file",
        type=str,
        default="datasets/ours/code/gen_reasoning_trace_prompt.md",
        help="The prompt template file for LLM generation."
    )
    parser.add_argument("--input_file",
        type=str,
        default=None,
        help="The input JSONL file containing necessary data fields to generate reasoning traces."
    )
    parser.add_argument("--output_file",
        type=str,
        default=None,
        help="JSONL file to save the generated reasoning traces."
    )

    # LLM generation parameters
    parser.add_argument("--model",
        type=str,
        default="hosted_vllm/gpt-oss-120b",
        help="LLM model to use for query generation."
    )
    parser.add_argument("--max_completion_tokens",
        type=int,
        default=1024*32,
        help="Max completion token"
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

    if args.action == "generate_reasoning_traces":
        generate_reasoning_traces(args)
    
    elif args.action == "convert_jsonl_to_md":
        convert_jsonl_to_md(args)
