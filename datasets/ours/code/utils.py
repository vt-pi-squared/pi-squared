import tiktoken
tiktoken_encoder = tiktoken.get_encoding("o200k_base")
count_tokens = lambda text: len(tiktoken_encoder.encode("".join(text), disallowed_special=()))

from collections import namedtuple
RetrievalResult = namedtuple("RetrievalResult", "doc_order chunk_order content")

import bm25s
import Stemmer  # optional: for stemming
stemmer = Stemmer.Stemmer("english")


class Corpus:
    """A corpus of docs. Indexes the docs on creation, normalizing the text beforehand with lemmatization."""
    def __init__(self, documents: list[dict], chunk_char_len: int):
        """
        :param documents: The list of evidences to index. Also support pre-chunked documents, in the format of list of list of str, where the inner list is the chunks for each document.
        :param chunk_char_len: The maximum length, in characters, of each chunk
        """
        self.documents = []
        for i, doc in enumerate(documents):
            if isinstance(doc, list):
                chunks = doc
            elif isinstance(doc, str):
                chunks = chunk_text(doc, max_chunk_size=chunk_char_len)
            for j, chunk in enumerate(chunks):
                self.documents.append(RetrievalResult(i, j, chunk))
        
        corpus = [doc.content for doc in self.documents]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25(method="bm25+", delta=1.5)
        retriever.index(corpus_tokens, leave_progress=False)
        self.index = retriever
    
    def best(self, query): #-> Iterable[RetrievalResult]:
        """Yield the best matching fragments to the given query."""
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        doc_ids, scores = self.index.retrieve(query_tokens, k=len(self.documents))
        for idx in doc_ids[0]:
            yield self.documents[idx]


def chunk_text(text, max_chunk_size=1024, chunk_on=("\n\n", "\n", ". "), chunker_i=0):
    """
    Recursively chunks *text* into a list of str, with each element no longer than *max_chunk_size*.
    Prefers splitting on the elements of *chunk_on*, in order.
    NOTE: no longer consider the following chunker ", ", " "
    """

    if len(text) <= max_chunk_size: # the chunk is small enough
        return [text]
    if chunker_i > len(chunk_on): # if no chunker left
        return [text]

    # split on the current character
    chunks = []
    split_char = chunk_on[chunker_i]
    for chunk in text.split(split_char):
        chunk = f"{chunk}{split_char}"
        if len(chunk) > max_chunk_size:  # this chunk needs to be split more, recurse
            chunks.extend(chunk_text(chunk, max_chunk_size, chunk_on, chunker_i + 1))
        elif chunks and len(chunk) + len(chunks[-1]) <= max_chunk_size:  # this chunk can be merged
            chunks[-1] += chunk
        else:
            chunks.append(chunk)

    # if the last chunk is just the split_char, yeet it
    if chunks[-1] == split_char:
        chunks.pop()

    # remove extra split_char from last chunk
    chunks[-1] = chunks[-1][: -len(split_char)]
    return chunks


def bm25plus_context_compaction(max_remaining_tokens, documents, main_instruction,
                                 chunk_char_len=1024, debug=False):
    """
    Compact context using BM25+ retrieval to fit within token limits.
    This function retrieves the most relevant chunks from documents 
    based on their similarity to the main instruction, 
    then selects chunks until reaching the maximum context length constraint.
    Selected chunks are reorganized by their original document and chunk order before being returned.
    
    Args:
        max_remaining_tokens: The maximum number of tokens allowed in the compacted context.
        documents: Either a single string or a list of strings representing the documents to compact.
            NOTE: `documents` now can be a list of list of chunked strings, 
                in which case BM25+ will be applied on the chunks directly.
        main_instruction: The instruction or query used to retrieve relevant chunks
            from the documents via BM25+ scoring.
        chunk_char_len: The character length for each chunk when splitting documents for BM25+ retrieval.
        debug: Whether to print debug information during compaction.
    Returns:
        str: A single compacted context string if input documents is a string.
        list[str]: A list of compacted context strings if input documents is a list,
            with one string per original document containing its selected chunks
            joined by "...\n" separator, ordered by original chunk positions.
    Example:
        Single document:
        >>> result = bm25plus_context_compaction(args, "long text...", "query")
        >>> result
        "compacted text ..."
        Multiple documents:
        >>> result = bm25plus_context_compaction(args, ["text1...", "text2..."], "query")
        >>> iresult
        ["compacted text from text1 ...", "compacted text from text2 ..."]
    """
    is_documents_list = isinstance(documents, list)
    if not is_documents_list:
        documents = [documents]

    # check if documents already fit within max_remaining_tokens, if so return as is
    total_tokens = sum(count_tokens(doc) for doc in documents)
    if total_tokens <= max_remaining_tokens:
        if debug:
            print(f"Total tokens in documents ({total_tokens}) is within the limit of {max_remaining_tokens}. No compaction needed.")
        return documents if is_documents_list else documents[0]
    
    # prepare corpus and get best matching chunks using BM25+
    if debug:
        print(f"compacting context for the main instruction\n```\n{main_instruction}\n```\n"
            f"up to {max_remaining_tokens} tokens using BM25+")
    
    corpus = Corpus(documents=documents, chunk_char_len=chunk_char_len)
    selected_chunks = []
    for chunk in corpus.best(main_instruction):
        num_tokens = count_tokens(chunk.content)
        max_remaining_tokens -= num_tokens
        if max_remaining_tokens < 0:
            break
        selected_chunks.append(chunk) # namedtuple("RetrievalResult", "doc_order chunk_order content")
        if debug:
            print(f"Picking chunk {chunk.doc_order}-{chunk.chunk_order} with {num_tokens} tokens")
    
    # reorganize selected_chunks by original document order and chunk order, and
    # return a chunked document for each original document
    # Get unique doc_order values and sort them
    unique_doc_orders = sorted(set(doc.doc_order for doc in selected_chunks))
    
    # For each doc_order, collect and join chunks by chunk_order
    reorganized_documents = ["" for _ in range(len(documents))]
    for doc_order in unique_doc_orders:
        chunks_for_doc = [doc for doc in selected_chunks if doc.doc_order == doc_order]
        chunks_for_doc.sort(key=lambda x: x.chunk_order)
        reorganized_documents[doc_order] = "...\n".join([chunk.content for chunk in chunks_for_doc])
    
    # Return based on input format
    if not is_documents_list:
        reorganized_documents = reorganized_documents[0]
    return reorganized_documents


if __name__ == "__main__":
    # test the bm25plus_context_compaction function
    documents = [
        "This is the first document. It contains some information about cats. Cats are cute.",
        "This is the second document. It contains some information about dogs. Dogs are loyal.",
        "This is the third document. It contains some information about birds. Birds can fly."
    ]
    main_instruction = "Tell me about cats and dogs."
    compacted_context = bm25plus_context_compaction(
        max_remaining_tokens=20,
        documents=documents,
        main_instruction=main_instruction,
        debug=True
    )
    print("\nCompacted Context:\n", compacted_context)
