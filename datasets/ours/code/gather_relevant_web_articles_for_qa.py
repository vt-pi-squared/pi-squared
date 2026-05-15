import os
import re
import requests
import json
import trafilatura
import func_timeout

SEARCH_CACHE_DIR = os.path.expanduser("~/.cache/websearch/queries/") # +{query}.json
WEB_CACHE_DIR = os.path.expanduser("~/.cache/webcontent/urls/") # +{normalize_url(url)}.txt
os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
os.makedirs(WEB_CACHE_DIR, exist_ok=True)

LangSearchAPIs = [
    "sk-9712e9995d324195a7817fe03accd92d",
    "sk-cf7e1f0547aa4a8fa7ca4038614efa69",
]


# ----------
# Searching
# ----------

# LangSearch https://docs.langsearch.com/api/web-search-api
@func_timeout.func_set_timeout(10)
def langsearch(query, api_key, count=10):
    """
    Web search using LangSearch API.
    Return: List of dict_keys(['id', 'name', 'url', 'displayUrl', 'snippet', 'summary', 'datePublished', 'dateLastCrawled'])
    """
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count
    })
    headers = {
        'Authorization': f'Bearer {api_key}', # free API key, feel free to use!
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    content = response.json()
    search_results = content["data"]["webPages"]["value"]
    return search_results


# OpenSerp https://github.com/karust/openserp
# curl command: curl "http://localhost:7000/mega/search?text={query}&engines=google,duckduckgo&limit={count}"
@func_timeout.func_set_timeout(10)
def openserp_search(query, count=10, host="http://localhost:7000", engines="google"):
    """
    Web search using OpenSerp package
    Return: List of dict_keys(['rank', 'url', 'title', 'description', ..])
    """
    url = f"{host.rstrip('/')}/mega/search"
    params = {"text": query, "engines": engines, "limit": count}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    search_results = resp.json()
    return search_results


def search(query, count=10):
    search_results = []
    basename = f"{normalize_url(query)}.txt"
    filepath = os.path.join(SEARCH_CACHE_DIR, basename)

    if os.path.exists(filepath):
        print(f"Loading cached search results \n\t from '{filepath}' for \n\t query '{query}'")
        with open(filepath, "r", encoding="utf-8") as f:
            search_results = json.load(f)
        return search_results
    else:
        try:
            for engine in ['duckduckgo', 'google']:
                print(f"Trying OpenSerp search with engine '{engine}' for query '{query}'")
                search_results = openserp_search(
                    query, count=count, engines=engine, host=f"http://localhost:{args.port}")
                if search_results:
                    break
            if not search_results:
                raise ValueError("Empty search results from OpenSerp")
        except:
            print(f"OpenSerp search failed for query '{query}' with error, fallback to LangSearch")
            for api_key in LangSearchAPIs:
                try:
                    search_results = langsearch(query, api_key, count=count)
                except Exception as e:
                    pass
                if search_results:
                    break
            if not search_results:
                print(f"LangSearch also failed for query '{query}', return empty search results")

        print(f"Caching search results \n\t to '{filepath}' for \n\t query '{query}'")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(search_results, f)
        return search_results


def demo():
    query = "The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) report"
    
    # langsearch
    search_results = langsearch(query, LangSearchAPIs[0], count=10)
    urls = [result["url"] for result in search_results]
    print(urls)
    # ['https://2024.emnlp.org/', 'https://2025.emnlp.org/', 'https://en.wikipedia.org/wiki/Empirical_Methods_in_Natural_Language_Processing', 'https://aclanthology.org/venues/emnlp/', 'https://2024.emnlp.org/program/', 'https://www.aclweb.org/portal/content/emnlp-2024-final-call-papers', 'https://2025.emnlp.org/calls/main_conference_papers/', 'https://www.cs.columbia.edu/tag/emnlp/', 'https://2024.emnlp.org/program/accepted_main_conference/', 'https://ai.engin.umich.edu/stories/eighteen-papers-by-cse-researchers-at-emnlp-2025']
    
    # openserp
    search_results = openserp_search(query)
    urls = [result["url"] for result in search_results]
    print(urls)


# ----------
# Crawling
# ----------

@func_timeout.func_set_timeout(15)
def trafilatura_crawl(url):
    """
    Crawl the webpage using Trafilatura package, return the text content.
    Maximize recall with html2txt function.
    Docs: https://trafilatura.readthedocs.io/en/latest/quickstart.html

    CLI: trafilatura -u https://2024.emnlp.org/ --markdown
    NOTE: Sections, paragraphs or tables are well separated with double new line char '\n\n' 
    """
    html = trafilatura.fetch_url(url)
    txt = trafilatura.extract(html, output_format="markdown", with_metadata=True, include_comments=False)
    # txt = trafilatura.html2txt(html) # pretty noisy
    return txt


def normalize_url(url):
    """
    Normalize the url to a valid filename, by replacing special characters with `-`.
    Also chunk the string up to 246 chars, as linux file basename length limit = 255 bytes
    """
    return re.sub(r'[^a-zA-Z0-9]', '-', url)[:246]


def cache_webpage(url):
    """
    Cache the content of the url to a local file, and return the basename of the cache file.
    """
    if not url:
        return None
    try:
        basename = f"{normalize_url(url)}.md"
        filepath = os.path.join(WEB_CACHE_DIR, basename)
        print(f"Caching webpage content \n\t to '{filepath}' for \n\t url '{url}'")
        if not os.path.exists(filepath):
            content = trafilatura_crawl(url)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        return basename
    except:
        print(f"\tError caching webpage for url '{url}'")
        return None


# ----------
# Get relevant web articles for QA
# ----------

def websearch_and_crawl(args, questions):
    if args.output_jsonl_file is not None:
        if os.path.exists(args.output_jsonl_file) and not args.debug:
            print(f"Loading JSONL file '{args.output_jsonl_file}'")
            fout = open(args.output_jsonl_file, "a", encoding="utf-8")
            # take the last non-empty line to get the last processed page title for resuming
            with open(args.output_jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    last_processed_question = None
                else:
                    last_row = lines[-1]
                    last_row = json.loads(last_row)
                    last_processed_question = last_row["question"]
                    print(f"Last processed question: {last_processed_question}")
        else:
            fout = open(args.output_jsonl_file, "w", encoding="utf-8")
            last_processed_question = None
    else:
        print("No jsonl file specified, skip this function")
        return

    # use the question string as question_id
    question_ids = [q['question'] for q in questions]
    n_questions = len(question_ids)
    if last_processed_question:
        start_index = question_ids.index(last_processed_question) + 1
        questions = questions[start_index:]
    else:
        start_index = 0

    for _, q in enumerate(questions, start_index+1):
        question = q['question']
        print("="*20 + f"\nQuestion {_}/{n_questions}: {question}:")

        # search
        search_results = search(question)
        urls = [result["url"] for result in search_results]

        # cache web content and get filepath
        file_basenames = [cache_webpage(url) for url in urls]
        file_basenames = [f for f in file_basenames if f is not None]

        q['relevant_web_urls'] = urls
        q['relevant_web_articles'] = file_basenames
        fout.write(json.dumps(q) + "\n")
        fout.flush()

        if args.debug:
            break

    print(f"Finished processing questions, output saved to '{args.output_jsonl_file}'")
    fout.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_qar_jsonl_file",
        type=str,
        default="datasets/ours/v2/spy_qar_ours_all.jsonl",
        help="The input JSONL file containing the QAR data. Each line includes keys table_id and question"
    )
    parser.add_argument("--port",
        type=int,
        default=7000,
        help="The port number for OpenSerp search API. Make sure the OpenSerp server is running and listening on this port."
    )
    parser.add_argument("--debug",
        action="store_true",
        help="Whether to run in debug mode (only process one table and print more logs)."
    )
    args = parser.parse_args()
    args.output_jsonl_file = args.input_qar_jsonl_file.replace(".jsonl", "_websearch.jsonl")
    qar = [json.loads(l) for l in open(args.input_qar_jsonl_file, "r").readlines()]

    questions = []
    for item in qar:
        if item.get('reasoning_traces', None):
            questions.append({"table_id": item["table_id"], "question": item["question"]})
            if args.debug:
                break

    websearch_and_crawl(args, questions)
