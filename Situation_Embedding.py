# Dual-embedding RAG with LangChain + FAISS
# - content index: embeddings of text chunks
# - header index: embeddings of flattened header path ("H1 > H2 > H3")
# - hybrid search: weighted sum of content + header similarities
#
# Requirements:
# pip install langchain faiss-cpu openai
# Set OPENAI_API_KEY or swap in another Embedding class.


import pandas as pd
import os
import json
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings       # swap if you use sentence-transformers
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

# ---------- config ----------
login("HF_TOKEN")
def load_dataframe(path_dir: str):
    df = pd.read_csv(path_dir)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
def build_indexes_from_text(df: pd.DataFrame,
                            embeddings=None,
                            max_words=700,
                            save_dir="FAISS/q_db"):
    """
    Build and save FAISS indexes + docs (JSON).
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")


    return faiss_content, faiss_header, docs_content

# ---------- hybrid retrieval ----------

def hybrid_search(query: str,
                  faiss_content: FAISS,
                  faiss_header: FAISS,
                  docs_content: List[Document],
                  k: int = 5,
                  w_content: float = 0.7,
                  w_header: float = 0.3):
    """
    Returns top-k Document objects (from content) ranked by hybrid score.
    Assumes similarity_search_with_score returns (Document, score) where lower score == closer (distance).
    We convert distance -> similarity by sim = 1/(1+distance).
    """
    assert faiss_content is not None
    # content results
    c_results = faiss_content.similarity_search_with_score(query, k=k)
    # header results
    h_results = faiss_header.similarity_search_with_score(query, k=k) if faiss_header else []

    scores = {}  # chunk_id -> combined similarity
    # content hits
    for doc, score in c_results:
        chunk_id = doc.metadata.get('chunk_id')
        sim = 1.0 / (1.0 + float(score))  # convert distance -> similarity
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_content * sim
    # header hits
    for doc, score in h_results:
        chunk_id = doc.metadata.get('chunk_id')
        sim = 1.0 / (1.0 + float(score))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_header * sim

    # Build sorted results
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    results = []
    for chunk_id, comb_sim in ranked[:k]:
        # find original doc
        doc = next((d for d in docs_content if d.metadata['chunk_id'] == chunk_id), None)
        if doc:
            doc.metadata['hybrid_score'] = comb_sim
            results.append(doc)
    return results

# ---------- example usage ----------
if __name__ == "__main__":
    # full_text should be your document as a single string
    with open("D:\\Code\\VME\\Enhanced Flow\\Trung Thu.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    build_indexes_from_text(full_text, save_dir = "Enhanced Flow/FAISS/full_db")
    # faiss_content, faiss_header, docs_content = load_indexes()
    # while True:
    #     q = input("Enter query (or 'exit'): ").strip()
    #     if q.lower() in ('exit', 'quit'):
    #         break
    #     if not q:
    #         continue
    #     out = hybrid_search(q, faiss_content, faiss_header, docs_content, k=4, w_content=0.3, w_header=0.7)
    #     for d in out:
    #         print("SCORE:", d.metadata.get('hybrid_score'), "HEADER:", d.metadata['header_path_str'])
    #         print(d.page_content[:300])
    #         print("---")
