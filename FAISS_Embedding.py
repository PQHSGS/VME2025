# Dual-embedding RAG with LangChain + FAISS
# - content index: embeddings of text chunks
# - header index: embeddings of flattened header path ("H1 > H2 > H3")
# - hybrid search: weighted sum of content + header similarities
#
# Requirements:
# pip install langchain faiss-cpu openai
# Set OPENAI_API_KEY or swap in another Embedding class.

from lib2to3.pgen2 import token
import re

import os
import json
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings       # swap if you use sentence-transformers
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

# ---------- config ----------
login("")
# ---------- parsing & chunking ----------

HEADER_RE = re.compile(r'^\s*((?:[A-Z]|(?:\d+(?:\.\d+)*)))\.\s+(.*)$')

def parse_hierarchy(text: str) -> List[Dict[str, Any]]:
    def get_level(token: str) -> int:
        if token.isalpha():   # "A", "B" ...
            return 1
        return token.count('.') + 2  # "1" -> 2, "1.1" -> 3

    lines = text.splitlines()
    stack = []  # each entry is (level, token, title, node_dict)
    nodes = []
    cur_node = None

    for i, line in enumerate(lines, 1):
        line = line.lstrip("\ufeff")
        m = HEADER_RE.match(line)
        if m:
            token = m.group(1).strip()
            title = m.group(2).strip()
            level = get_level(token)
            print(f"[line {i}] HEADER token={token!r} title={title!r} level={level}")
            if cur_node:
                print(f"  closing node {cur_node['token']}")
                nodes.append(cur_node)
            while stack and stack[-1][0] >= level:
                popped = stack.pop()
                print(f"  pop stack {popped[1]!r}")
            path = [p[2] for p in stack] + [title]
            node = {
                'token': token,
                'title': title,
                'level': level,
                'path': path,
                'content': ''
            }
            print(f"  new node path={path}")
            stack.append((level, token, title, node))
            cur_node = node
        else:
            if cur_node:
                if cur_node['content']:
                    cur_node['content'] += '\n' + line
                else:
                    cur_node['content'] = line
                print(f"[line {i}] append content to {cur_node['token']!r}")
            else:
                print(f"[line {i}] preamble: {line!r}")
                pass
    if cur_node:
        print(f"final closing node {cur_node['token']}")
        nodes.append(cur_node)
    return nodes

def chunk_text(text: str, max_words: int = 700) -> List[str]:
    """
    Simple sentence/newline splitter + accumulate to enforce max_words per chunk.
    """
    if not text or not text.strip():
        return []
    # split by sentences or newlines
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', text.strip())
    chunks = []
    cur = []
    cur_words = 0
    for p in parts:
        pw = len(p.split())
        if cur_words + pw > max_words and cur:
            chunks.append(' '.join(cur).strip())
            cur = [p]
            cur_words = pw
        else:
            cur.append(p)
            cur_words += pw
    if cur:
        chunks.append(' '.join(cur).strip())
    return chunks

# ---------- build LangChain Documents for both indexes ----------

def build_indexes_from_text(full_text: str,
                            embeddings=None,
                            max_words=700,
                            save_dir="FAISS/full_db"):
    """
    Build and save FAISS indexes + docs (JSON).
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")

    nodes = parse_hierarchy(full_text)
    docs_content, docs_header = [], []
    chunk_id = 0
    for node in nodes:
        header_path_str = " > ".join(node['path'])
        for chunk in chunk_text(node.get('content', ''), max_words=max_words):
            meta = {
                'chunk_id': str(chunk_id),
                'header_path': node['path'],
                'header_path_str': header_path_str,
                'token': node['token'],
                'title': node['title'],
            }
            docs_content.append(Document(page_content=chunk, metadata=meta))
            docs_header.append(Document(page_content=header_path_str,
                                        metadata={'chunk_id': str(chunk_id)}))
            chunk_id += 1

    faiss_content = FAISS.from_documents(docs_content, embeddings) if docs_content else None
    faiss_header = FAISS.from_documents(docs_header, embeddings) if docs_header else None

    # ---------- save ----------
    os.makedirs(save_dir, exist_ok=True)
    if faiss_content:
        faiss_content.save_local(os.path.join(save_dir, "content_index"))
    if faiss_header:
        faiss_header.save_local(os.path.join(save_dir, "header_index"))

    # save docs_content to JSON
    ctx_path = os.path.join(save_dir, "context.json")
    docs_json = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs_content
    ]
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(docs_json, f, ensure_ascii=False, indent=2)

    return faiss_content, faiss_header, docs_content


def load_indexes(embeddings=None, load_dir="FAISS/full_db"):
    """
    Load FAISS indexes + docs (from JSON).
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")

    faiss_content, faiss_header, docs_content = None, None, None

    content_path = os.path.join(load_dir, "content_index")
    header_path = os.path.join(load_dir, "header_index")
    ctx_path = os.path.join(load_dir, "context.json")

    if os.path.exists(content_path):
        faiss_content = FAISS.load_local(content_path, embeddings, allow_dangerous_deserialization=True)
    if os.path.exists(header_path):
        faiss_header = FAISS.load_local(header_path, embeddings, allow_dangerous_deserialization=True)
    if os.path.exists(ctx_path):
        with open(ctx_path, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
        docs_content = [Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in docs_raw]

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
