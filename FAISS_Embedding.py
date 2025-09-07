# pipeline_hybrid_final.py
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

# ---------- helpers ----------
def extract_title_and_keywords(raw: str):
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", raw.strip())
    if m:
        title = m.group(1).strip()
        kws = [k.strip() for k in re.split(r"[,;]", m.group(2)) if k.strip()]
        return title, kws
    return raw.strip(), []

# ---------- parsing ----------
def parse_trungthu(path):
    fields = {}
    sections = {}
    cur_field_id = cur_field_title = None
    cur_section_id = None
    buf = []
    field_re = re.compile(r"^(\d+)\.\s*(.+)")
    section_re = re.compile(r"^(\d+\.\d+)\.\s*(.+)")

    def flush_section():
        nonlocal buf, cur_section_id
        if cur_section_id and buf:
            sections[cur_section_id]["content"] = "\n".join(buf).strip()
            buf = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip():
                continue
            m_field = field_re.match(line)
            m_sec = section_re.match(line)
            if m_field and not m_sec:
                flush_section()
                num, raw_title = m_field.group(1), m_field.group(2)
                cur_field_id = f"F:{num}"
                title, kws = extract_title_and_keywords(raw_title)
                cur_field_title = title
                fields[cur_field_id] = {"field_id": cur_field_id, "field_title": title, "keywords": kws, "sections": []}
                cur_section_id = None
                buf = []
                continue
            if m_sec:
                flush_section()
                num, raw_title = m_sec.group(1), m_sec.group(2)
                cur_section_id = f"S:{num}"
                title, kws = extract_title_and_keywords(raw_title)
                sections[cur_section_id] = {
                    "section_id": cur_section_id,
                    "section_title": title,
                    "field_id": cur_field_id,
                    "field_title": cur_field_title,
                    "keywords": kws,
                    "content": ""
                }
                # register section under field
                if cur_field_id:
                    fields.setdefault(cur_field_id, {"field_id": cur_field_id, "field_title": cur_field_title, "keywords": [], "sections": []})
                    fields[cur_field_id]["sections"].append(cur_section_id)
                buf = []
                continue
            if cur_section_id:
                buf.append(line)
    flush_section()
    return fields, sections

def parse_questions(path):
    data = []
    cur_field_id = cur_field = cur_section_id = cur_section = None
    field_re, sec_re = re.compile(r"^(\d+)\.\s*(.+)"), re.compile(r"^(\d+\.\d+)\.\s*(.+)")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m_field = field_re.match(line)
            m_sec = sec_re.match(line)
            if m_field and not m_sec:
                num, title = m_field.group(1), m_field.group(2)
                cur_field_id = f"F:{num}"
                cur_field = f"{num}. {title.strip()}"
                cur_section_id = cur_section = None
                continue
            if m_sec:
                num, title = m_sec.group(1), m_sec.group(2)
                cur_section_id = f"S:{num}"
                cur_section = f"{num}. {title.strip()}"
                continue
            if line.startswith("-") and cur_section_id:
                q = line.lstrip("-").strip()
                data.append({"field_id": cur_field_id, "field_title": cur_field, "section_id": cur_section_id, "section_title": cur_section, "question": q})
    return data

# ---------- indexing ----------
def build_and_save(texts, metas, outdir):
    os.makedirs(outdir, exist_ok=True)
    vecs = np.array(EMB.embed_documents(texts), dtype="float32")
    faiss.normalize_L2(vecs)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    ids = [str(i) for i in range(len(texts))]
    docstore = InMemoryDocstore({ids[i]: Document(page_content=texts[i], metadata=metas[i]) for i in range(len(texts))})
    vs = FAISS(embedding_function=EMB, index=index, docstore=docstore, index_to_docstore_id={i: ids[i] for i in range(len(texts))})
    vs.normalize_L2 = True
    vs.save_local(outdir)
    return vs

def build_all():
    fields, sections = parse_trungthu(SRC_TEXT)
    qs = parse_questions(SRC_QUEST)

    # anchors: field titles, section titles, each keyword individually
    a_texts = []
    a_metas = []
    # fields
    for fid, f in fields.items():
        a_texts.append(f["field_title"])
        a_metas.append({"type": "field", "field_id": fid, "field_title": f["field_title"]})
        for kw in f.get("keywords", []):
            a_texts.append(kw)
            a_metas.append({"type": "keyword", "keyword": kw, "field_id": fid, "section_id": None})
    # sections
    for sid, s in sections.items():
        a_texts.append(s["section_title"])
        a_metas.append({"type": "section", "section_id": sid, "field_id": s["field_id"], "section_title": s["section_title"], "field_title": s["field_title"]})
        for kw in s.get("keywords", []):
            a_texts.append(kw)
            a_metas.append({"type": "keyword", "keyword": kw, "section_id": sid, "field_id": s["field_id"]})
    # content index: one doc per section (full text)
    c_texts = []
    c_metas = []
    for sid, s in sections.items():
        c_texts.append(s.get("content", ""))
        c_metas.append({"type": "content", "section_id": sid, "field_id": s["field_id"], "section_title": s["section_title"], "field_title": s["field_title"]})
    build_and_save(c_texts, c_metas, ANCHOR_DIR)

    # questions index: one doc per question
    q_texts = []
    q_metas = []
    for q in qs:
        q_texts.append(q["question"])
        q_metas.append({"type": "question", "question": q["question"], "section_id": q["section_id"], "field_id": q["field_id"], "section_title": q["section_title"], "field_title": q["field_title"]})
    build_and_save(q_texts, q_metas, QUESTIONS_DIR)
    # save context
    ctx = {"fields": fields, "sections": sections}
    with open(CTX_PATH, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

build_all()

