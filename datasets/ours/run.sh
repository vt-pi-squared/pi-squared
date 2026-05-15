conda activate pisquared
mkdir -p datasets/ours/v2


# Step 0: Prepare the list of wiki pages to crawl.
python datasets/ours/code/v2_gather_wiki_pages.py
# ---> Resulting in `datasets/ours/v2/collected_wiki_pages.txt`


# Step 1: Crawl tables and metadata from Wikipedia
python datasets/ours/code/wikipedia_table_crawler.py \
    --action crawl \
    --page_titles_file datasets/ours/v2/collected_wiki_pages.txt \
    --output_jsonl_file "datasets/ours/v2/tab_doc_expansion.jsonl" \
# ---> Resulting in datasets/ours/v2/tab_doc_expansion_filtered.jsonl"


# Step 2: Generate QAR with wiki context. Powered by "Qwen3.5-35B-A3B-FP8"
export MODEL="Qwen3.5-35B-A3B-FP8"
python datasets/ours/code/gen_code_qa_reasoning_trace.py \
    --action generate_query \
    --input_tab_doc_jsonl_file "datasets/ours/v2/tab_doc_expansion_filtered.jsonl" \
    --output_jsonl_file "datasets/ours/v2/spy_qar_ours_all.jsonl" \
    --table_source ours_all \
    --model "hosted_vllm/${MODEL}" \
# ---> Resulting in datasets/ours/v2/spy_qar_ours_all_filtered.jsonl"


# Step 3: Run web search to gather relevant realistic documents
python datasets/ours/code/gather_relevant_web_articles_for_qa.py \
    --input_qar_jsonl_file datasets/ours/v2/spy_qar_ours_all_filtered.jsonl \
# ---> Resulting in datasets/ours/v2/spy_qar_ours_all_filtered_websearch.jsonl"


# Step 4: Post-process and merge table, QA, and web search data. Chunk the context to 96K tokens
python datasets/ours/code/v2_merge_and_postprocess_ctqar.py \
    --tab_doc_file datasets/ours/v2/tab_doc_expansion_filtered.jsonl \
    --qar_file datasets/ours/v2/spy_qar_ours_all_filtered.jsonl \
    --websearch_file datasets/ours/v2/spy_qar_ours_all_filtered_websearch.jsonl \
    --merged_file datasets/ours/v2/ctqar_merged.jsonl \
    --output_file datasets/ours/v2/ctqar_clean_context.jsonl


# Step 5: Generate reasoning traces.
export MODEL="Qwen3.5-35B-A3B-FP8"
python datasets/ours/code/gen_reasoning_trace_w_llm.py \
    --input_file datasets/ours/v2/ctqar_clean_context.jsonl \
    --output_file datasets/ours/v2/ctqar_clean_llm_rt_${MODEL}.jsonl

# Step 5.5: Convert a few samples into in Markdown format for inspection
python datasets/ours/code/gen_reasoning_trace_w_llm.py \
    --action convert_jsonl_to_md \
    --output_file datasets/ours/v2/ctqar_llm_rt_${MODEL}.jsonl

