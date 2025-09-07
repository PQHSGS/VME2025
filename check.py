# pipeline_hybrid.py
import os, re, json, numpy as np, faiss
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

# ---------- config ----------
login("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")
SRC_TEXT = "Trung Thu.txt"
SRC_QUEST = "Questions.txt"
OUT_DIR = "FAISS/HyPE_anchors"
ANCHOR_DIR = os.path.join(OUT_DIR, "anchors")
QUESTIONS_DIR = os.path.join(OUT_DIR, "questions")
CTX_PATH = os.path.join(OUT_DIR, "context.json")
os.makedirs(OUT_DIR, exist_ok=True)

EMB_MODEL = "hiieu/halong_embedding"
EMB = HuggingFaceEmbeddings(model_name=EMB_MODEL)


# ---------- retrieval ----------
def load_store():
    vs_a = FAISS.load_local(ANCHOR_DIR, EMB, allow_dangerous_deserialization=True); vs_a.normalize_L2 = True
    vs_q = FAISS.load_local(QUESTIONS_DIR, EMB, allow_dangerous_deserialization=True); vs_q.normalize_L2 = True
    with open(CTX_PATH, "r", encoding="utf-8") as f:
        ctx = json.load(f)
    return vs_a, vs_q, ctx

def retrieve(query: str, topk_anchors: int = 5, topk_question: int = 5):
    vs_a, vs_q, ctx = load_store()
    a_hits = vs_a.similarity_search_with_score(query, k=topk_anchors)
    q_hits = vs_q.similarity_search_with_score(query, k=topk_question)

    best_scores = {}  # section_id -> best score
    def bump(sid, score):
        if not sid: return
        best_scores[sid] = max(score, best_scores.get(sid, -1e9))

    # anchors -> map to section ids
    for doc, score in a_hits:
        meta = doc.metadata
        typ = meta.get("type")
        if typ == "field":
            fid = meta.get("field_id")
            for sid in ctx["fields"].get(fid, {}).get("sections", []):
                bump(sid, float(score))
        elif typ == "section":
            sid = meta.get("section_id")
            bump(sid, float(score))
        elif typ == "keyword":
            sid = meta.get("section_id")
            if sid:
                bump(sid, float(score))
            else:
                fid = meta.get("field_id")
                for sid in ctx["fields"].get(fid, {}).get("sections", []):
                    bump(sid, float(score))
        elif typ == "context":
            sid = meta.get("section_id")
            bump(sid, float(score))
    for doc, score in q_hits:
        sid = doc.metadata.get("section_id")
        bump(sid, float(score))

    # content hits -> direct
    for doc, score in q_hits:
        sid = doc.metadata.get("section_id")
        bump(sid, float(score))

    if not best_scores:
        return ""  # empty if nothing matched

    # sort sections by score desc
    sorted_sections = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

    # build output text (field -> section -> content)
    out_blocks = []
    for sid, sc in sorted_sections:
        sec = ctx["sections"].get(sid)
        if not sec:
            continue
        fid = sec["field_id"]
        field = ctx["fields"].get(fid, {})
        title_line = f"{field.get('field_title','')} -> {sec.get('section_title','')}"
        content = sec.get("content","").strip()
        out_blocks.append(f"{title_line}\n{content}\n[score={sc:.3f}]")

    return "\n\n-----\n\n".join(out_blocks)

# ---------- run ----------
if __name__ == "__main__":
    vs_a, vs_q, ctx = load_store()
    while True:
        q = input("Query (empty to quit): ").strip()
        if not q:
            break
        print(retrieve(q))