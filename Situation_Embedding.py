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
login("")
def load_dataframe(path_dir: str):
    df = pd.read_csv(path_dir)
    df.fillna('', inplace=True)
    print(df['Câu hỏi'])
    return df
def build_indexes_from_questions(load_dir: str,
                            embeddings=None,
                            save_dir="FAISS/q_db"):
    """
    Build and save FAISS indexes + docs (JSON).
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
    df = load_dataframe(load_dir)
    questions_content= []
    for i in range(len(df)):
        meta = {
            'question': df['Câu hỏi'][i],
            'guidance': df['Hướng dẫn'][i],
            'answer': df['Câu trả lời mẫu'][i],
        }
        questions_content.append(Document(page_content=df['Câu hỏi'][i], metadata=meta))
    faiss_content = FAISS.from_documents(questions_content, embeddings)
    # save
    os.makedirs(save_dir, exist_ok=True)
    faiss_content.save_local(os.path.join(save_dir, "question_index"))
    with open(os.path.join(save_dir, "questions.json"), "w", encoding="utf-8") as f:
        json.dump([doc.metadata for doc in questions_content], f, ensure_ascii=False, indent=2)
def load_indexes(embeddings=None, load_dir="FAISS/full_db"):
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
    path = os.path.join(load_dir, "question_index")
    question_index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    with open(os.path.join(load_dir, "questions.json"), "r", encoding="utf-8") as f:
        questions = json.load(f)
    return question_index, questions

# ---------- hybrid retrieval ----------

def search(query: str, faiss_question: FAISS, k=3):
    results=[]
    res = faiss_question.similarity_search_with_score(query, k=k)
    for r in res:
        doc, score = r
        doc.metadata['score'] = score
        results.append(doc)
    return results

# ---------- example usage ----------
if __name__ == "__main__":
    build_indexes_from_questions("D:\\Code\\VME\\Enhanced Flow\\Questions.csv", save_dir = "FAISS/full_db")
    question_index, questions = load_indexes()
    while True:
        q = input("Enter query (or 'exit'): ").strip()
        if q.lower() in ('exit', 'quit'):
            break
        if not q:
            continue
        out = search(q, question_index, k=4)
        for d in out:
            print(f"Score: {d.metadata['score']:.4f}")
            print(f"Question: {d.metadata['question']}")
            print(f"Guidance: {d.metadata['guidance']}")
            print(f"Answer: {d.metadata['answer']}")
            print("-" * 40)