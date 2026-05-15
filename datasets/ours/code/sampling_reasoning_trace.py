"""
Reasoning Trace Generation Pipeline

Generates high-quality reasoning trajectories for question-answering tasks using the Rejection Sampling technique 
(validate the real reasoning traces with ground-truth answer).
"""

import json, copy
from tqdm import tqdm
from helper import *
from dspy_program_qar_gen import *


# long_context_qa_description = """
# You are a helpful assistant. You will answer a question based on the given context.
# If the timestamp of different information in the context is provided, you should consider it when answering the question.

# Please concisely 1) extract and aggregate necessary information from the context to answer the question, then 2) reason step by step to arrive at the final answer.
# """
long_context_qa_description = open("datasets/ours/code/sampling_reasoning_trace_prompt.md", "r").read()

class LongContextQA(dspy.Signature):
    __doc__ = long_context_qa_description
    context: str = dspy.InputField(desc='The context provided to answer the question, can be a table, a long span of text, or multiple retrieved documents')
    question: str = dspy.InputField(desc='The original question being answered')
    extraction_and_reasoning: str = dspy.OutputField(desc='The intermediate extraction (if applicable) and reasoning steps to arrive at the final answer, which should be in a step-by-step format')
    final_answer: str = dspy.OutputField(desc='The final answer to the question based on the context, which should be a concise answer without including the reasoning steps')

LongContextQA_parser = lambda text: dspy.ChatAdapter().parse(LongContextQA, text)

def dspy_simple_format_correction(text):
    for incorr, corr in [('[[##', '[[ ##'), ('##]]', '## ]]'), ('[[# ', '[[ ##'), (' #]]', '## ]]')]:
        text = text.replace(incorr, corr)
    return text


def generate_reasoning_traces(args):
    all_samples = [json.loads(l) for l in open(args.input_file, 'r', encoding='utf-8').readlines() if l.strip()]
    n_samples = len(all_samples)
    identifier_key = 'question'
    
    fout, start_index = prepare_and_load_cache(args.output_file, all_samples, identifier_key)
    all_samples = all_samples[start_index:]
    
    for _, sample in tqdm(enumerate(all_samples, start_index+1), total=len(all_samples)):
        sample.pop('llm_generated_reasoning_traces', '')
        try:
            print("="*20 + f"\nSample {_}/{n_samples}: {sample[identifier_key]}")
            messages = get_messages_templates(LongContextQA)
            messages[-1]["content"] = messages[-1]["content"].format(
                context=sample.get(args.context_key),
                question=sample.get("question"),
            )
            response, thinking_content = llm_call(args, messages, get_thinking_tokens=True)
            if response:
                parsed_response = LongContextQA_parser(dspy_simple_format_correction(response))
                dfc_messages = get_messages_templates(DfComparison)
                dfc_messages[-1]["content"] = dfc_messages[-1]["content"].format(
                    question=sample["question"],
                    metadata=sample["metadata"],
                    result1=simplify_to_md_or_str(sample["final_answer_sql"]),
                    result2=simplify_to_md_or_str(parsed_response["final_answer"])
                )
                comparison_response = llm_call(args=args, messages=dfc_messages)
                comparison_response = DfComparison_parser(comparison_response)
                sample['llm_generated_reasoning_traces'] = response
                sample['llm_thinking_content'] = thinking_content
                sample['answer_comparison'] = comparison_response["judgment"]
                fout.write(json.dumps(sample) + "\n")
                fout.flush()
            else:
                print(f"\tError generating reasoning trace for sample with {identifier_key} '{sample[identifier_key]}'")
            if args.debug:
                print(f"\t[DEBUG] Messages sent to LLM:\n...{messages[-1]["content"][-1000:]}")
                print(f"\t[DEBUG] Generated reasoning trace:\n{response}")
                print(f"\t[DEBUG] Finish processing the first sample, exiting due to debug mode.")
                break
        except Exception as e:
            print(f"\tError processing sample with {identifier_key} '{sample[identifier_key]}': {e}")
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
            f"## Condition Column:\n{item['condition_col']}",
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


def filter_reasoning_traces(args):
    """
    This function load 10 files of sampled reasoning traces, and
    - filter out reasoning traces with incorrect answer (~ wrong reasoning traces)
    - filter out questions with X% correct answers across different attempts (~ easy samples)
    """    
    attempts_context = [
        [json.loads(l) for l in open(file, 'r').readlines() if l.strip()]
        for file in [f"datasets/ours/v2/ctqar_clean_llm_rt_context_{i}_gpt-oss-20b.jsonl" for i in range(1, 6)]
    ]
    attempts_df_md = [
        [json.loads(l) for l in open(file, 'r').readlines() if l.strip()]
        for file in [f"datasets/ours/v2/ctqar_clean_llm_rt_df_md_{i}_gpt-oss-20b.jsonl" for i in range(1, 6)]
    ]
    attempts =  attempts_df_md + attempts_context

    # Group by question
    group_by_question = {}
    for attempt in attempts:
        for sample in attempt:
            group_by_question.setdefault(sample.get('question'), []).append(sample.get('answer_comparison'))

    for q in group_by_question:
        comparisons = group_by_question[q]
        total = len(comparisons)
        match_count = sum(comparisons)
        # print(f"Question: {q}\nTotal Attempts: {total}, Match Count: {match_count}, Match Rate: {match_count/total:.2f}\n")

    # Filter: keep only filtered_questions with correct answers, skip easy questions
    filtered_questions = []
    all_failed_questions = []
    for q, comparisons in group_by_question.items():
        total = len(comparisons)
        match_count = sum(comparisons)
        if match_count == 0:
            # print(f"Question: {q}\nAll attempts are incorrect, likely due to wrong reasoning trace. Consider filtering out this question.\n")
            all_failed_questions.append(q)
        elif match_count / total > 0.8:
            # print(f"Question: {q}\nMost attempts are correct (match rate {match_count/total:.2f}), likely an easy question. Consider filtering out this question.\n")
            pass
        else:
            filtered_questions.append(q)
    print(f"Total hard questions after filtering based on student performance: {len(filtered_questions)}, and {len(all_failed_questions)} failed questions.\n")

    # Gather correct reasoning traces for the filtered questions
    # for each question, take at most one correct reasoning trace for each `context` and `df_md` setting
    filtered_reasoning_traces_context = []
    filtered_questions_context = copy.deepcopy(filtered_questions)
    for attempt in attempts_context:
        for sample in attempt:
            if sample.get('question') in filtered_questions_context and sample.get('answer_comparison'):
                filtered_questions_context.remove(sample.get('question'))
                filtered_reasoning_traces_context.append({
                    'table_id': sample.get('table_id'),
                    'metadata': sample.get('metadata'),
                    'question': sample.get('question'),
                    'context': sample.get('context'),
                    'final_answer_sql': sample.get('final_answer_sql'),
                    'llm_generated_reasoning_traces': sample.get('llm_generated_reasoning_traces'),
                    'llm_thinking_content': sample.get('llm_thinking_content'),
                })
    filtered_reasoning_traces_df_md = []
    filtered_questions_df_md = copy.deepcopy(filtered_questions)
    for attempt in attempts_df_md:
        for sample in attempt:
            if sample.get('question') in filtered_questions_df_md and sample.get('answer_comparison'):
                filtered_questions_df_md.remove(sample.get('question'))
                filtered_reasoning_traces_df_md.append({
                    'table_id': sample.get('table_id'),
                    'metadata': sample.get('metadata'),
                    'question': sample.get('question'),
                    'context': sample.get('df_md'),
                    'final_answer_sql': sample.get('final_answer_sql'),
                    'llm_generated_reasoning_traces': sample.get('llm_generated_reasoning_traces'),
                    'llm_thinking_content': sample.get('llm_thinking_content'),
                })

    # Save the filtered reasoning traces to a new JSONL file
    with open("datasets/ours/v2/ctqar_clean_llm_rt_mixed_filtered_gpt-oss-20b.jsonl", "w") as fout:
        for trace in filtered_reasoning_traces_context+filtered_reasoning_traces_df_md:
            fout.write(json.dumps(trace) + "\n")
    print(f"Total filtered questions: {len(filtered_questions)}.\n"
          f"Total filtered reasoning traces with free text context saved: {len(filtered_reasoning_traces_context)}.\n"
          f"Total filtered reasoning traces with table context saved: {len(filtered_reasoning_traces_df_md)}.")



def filter_reasoning_traces_teacher(args):
    """
    Filter reasoning traces based on student's performance (for hard question) and teacher's demonstration (correct one).
    """
    attempts_student = [
        [json.loads(l) for l in open(file, 'r').readlines() if l.strip()]
        for file in [f"datasets/ours/v2/ctqar_clean_llm_rt_context_{i}_gpt-oss-20b.jsonl" for i in range(1, 6)]
    ]
    attempts_teacher = [
        [json.loads(l) for l in open(file, 'r').readlines() if l.strip()]
        for file in [f"datasets/ours/v2/ctqar_clean_llm_rt_rs_context_1_gpt-oss-120b{_}.jsonl" for _ in ["", "_high"]]
    ]

    # Group by question
    group_by_question = {}
    for attempt in attempts_student:
        for sample in attempt:
            group_by_question.setdefault(sample.get('question'), []).append(sample.get('answer_comparison'))

    for q in group_by_question:
        comparisons = group_by_question[q]
        total = len(comparisons)
        match_count = sum(comparisons)
        # print(f"Question: {q}\nTotal Attempts: {total}, Match Count: {match_count}, Match Rate: {match_count/total:.2f}\n")

    # Filter: keep only filtered_questions with correct answers, skip easy questions
    filtered_questions = []
    all_failed_questions = []
    for q, comparisons in group_by_question.items():
        total = len(comparisons)
        match_count = sum(comparisons)
        if match_count / total > 4/5:
            # print(f"Question: {q}\nMost attempts of students are correct (match rate {match_count/total:.2f}), likely an easy question. Consider filtering out this question.\n")
            pass
        elif match_count == 0:
            # print(f"Question: {q}\nAll attempts of students are incorrect, likely due to wrong reasoning trace or very hard question. Consider filtering out this question.\n")
            all_failed_questions.append(q)
            filtered_questions.append(q)
        else:
            filtered_questions.append(q)
    print(f"Total hard questions after filtering based on student performance: {len(filtered_questions)}, amond them {len(all_failed_questions)} failed questions.\n")

    # Gather correct reasoning traces for the filtered questions
    filtered_reasoning_traces = []
    filtered_questions_context = copy.deepcopy(filtered_questions)
    for attempt in attempts_teacher:
        for sample in attempt:
            if sample.get('question') in filtered_questions_context and sample.get('answer_comparison'):
                filtered_reasoning_traces.append({
                    'table_id': sample.get('table_id'),
                    'metadata': sample.get('metadata'),
                    'question': sample.get('question'),
                    'context': sample.get('context'),
                    'final_answer_sql': sample.get('final_answer_sql'),
                    'llm_generated_reasoning_traces': sample.get('llm_generated_reasoning_traces'),
                    'llm_thinking_content': sample.get('llm_thinking_content'),
                })

    # Save the filtered reasoning traces to a new JSONL file
    with open("datasets/ours/v2/ctqar_clean_llm_rt_mixed_filtered_gpt-oss-120b.jsonl", "w") as fout:
        for trace in filtered_reasoning_traces:
            fout.write(json.dumps(trace) + "\n")
    print(f"Total filtered questions: {len(filtered_questions)}.\n"
          f"Total filtered traces from gpt-oss-120b saved: {len(filtered_reasoning_traces)}.\n")


def filter_dedicated_reasoning_traces(args):
    """
    Filter dedicated reasoning traces based on the correctness of demonstrations.
    """
    for tc_size in ["20b", "120b"]:
        for re in ["", "_high"]: # low or high
            filename = f"datasets/ours/v2/ctqar_clean_dedicated_rs_context_1_gpt-oss-{tc_size}{re}.jsonl"
            attempt_teacher = [json.loads(l) for l in open(filename, 'r').readlines() if l.strip()]
            filtered_reasoning_traces = []
            for sample in attempt_teacher:
                if sample.get('answer_comparison'):
                    filtered_reasoning_traces.append({
                    'table_id': sample.get('table_id'),
                    'metadata': sample.get('metadata'),
                    'question': sample.get('question'),
                    'context': sample.get('context'),
                    'final_answer_sql': sample.get('final_answer_sql'),
                    'llm_generated_reasoning_traces': sample.get('llm_generated_reasoning_traces'),
                    'llm_thinking_content': sample.get('llm_thinking_content'),
                })

            # Print out some statistics
            data = filtered_reasoning_traces
            reasoning_traces_token_count = [count_tokens(item["llm_generated_reasoning_traces"]) for item in data]
            thinking_token_count = [count_tokens(item["llm_thinking_content"]) for item in data]
            print(f"Reasoning traces token count for gpt-oss-{tc_size}{re}:\n",
                  pd.Series(reasoning_traces_token_count).describe().to_dict())
            print(f"Thinking token count for gpt-oss-{tc_size}{re}:\n",
                  pd.Series(thinking_token_count).describe().to_dict())

            # Save the filtered reasoning traces to a new JSONL file
            with open(f"datasets/ours/v2/ctqar_clean_dedicated_rs_filtered_gpt-oss-{tc_size}{re}.jsonl", "w") as fout:
                for trace in filtered_reasoning_traces:
                    fout.write(json.dumps(trace) + "\n")
            print(f"Total traces from gpt-oss-{tc_size}{re} saved: {len(filtered_reasoning_traces)}.\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--action",
        type=str,
        default="generate_reasoning_traces",
        help="Action to perform."
    )
    parser.add_argument("--debug",
        action="store_true",
        help="Whether to run in debug mode (only process one table and print more logs)."
    )

    # Reasoning trace generation settings
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
    parser.add_argument("--context_key",
        type=str,
        default="context",
        choices=["context", "df_md"],
        help="The key in the JSON object that represents the context."
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
    parser.add_argument("--api_key",
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

    elif args.action == "filter_rejection_sampling":
        filter_reasoning_traces(args)

    elif args.action == "filter_rejection_sampling_teacher":
        filter_reasoning_traces_teacher(args)

    elif args.action == "filter_dedicated_reasoning_traces":
        filter_dedicated_reasoning_traces(args)
