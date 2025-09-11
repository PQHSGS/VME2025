import os
import json
import time
from flask import Flask, request, jsonify, abort
from typing import Dict
from uuid import uuid4

# Import LangChain components.
# These libraries are used for building the conversational agent.
from langchain_core.tools import Tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# Set environment variables for LangSmith and Google APIs.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_PROJECT'] = 'vme2025_gemini_embedding_004'
os.environ['GOOGLE_API_KEY'] = "AIzaSyAxL30FKtRAXzcThLyTQf9WN75VsVfCHkA"

# Initialize the Flask application instance.
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Store conversation memories per user session.
user_sessions: Dict[str, ConversationBufferMemory] = {}

#FAISS setup
OUT_DIR = "FAISS/HyPE_anchors"
ANCHOR_DIR = os.path.join(OUT_DIR, "anchors")
QUESTIONS_DIR = os.path.join(OUT_DIR, "questions")
CTX_PATH = os.path.join(OUT_DIR, "context.json")

EMB_MODEL = "hiieu/halong_embedding"
EMB = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vs_a = FAISS.load_local(ANCHOR_DIR, EMB, allow_dangerous_deserialization=True)
vs_q = FAISS.load_local(QUESTIONS_DIR, EMB, allow_dangerous_deserialization=True)
# load context.json (fields + sections)
ctx_path = os.path.join(OUT_DIR, "context.json")
with open(ctx_path, "r", encoding="utf-8") as f:
    ctx = json.load(f)

@tool
def retrieve(query: str, topk_anchors: int = 4, topk_question: int = 4):
    """Retrieve relevant sections from the FAISS vector store based on the query."""
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

# Define the Agent and Tools outside the endpoint function.
# This ensures they are initialized only once when the server starts.
tools = [
    Tool(
        name="museum_documents_tool",
        func=retrieve,
        description="Trích xuất thông tin liên quan từ tài liệu nội bộ."
    )
]

# Initialize the large language model.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print("Error: system_prompt.txt file not found.")
    abort(500, description="system_prompt.txt file not found.")

# API endpoint for chatting with the agent.
@app.route("/chat", methods=["POST"])
def chat_with_agent():
    data = request.get_json()
    if not data or 'data' not in data:
        return jsonify({"error": "Request body must be JSON with a 'data' field."}), 400

    request_data = data.get('data')
    request_id = data.get('id')

    if request_id and request_id in user_sessions:
        memory = user_sessions[request_id]
        user_id = request_id
        print(f"Found existing session for ID: {user_id}")
    else:
        user_id = request_id if request_id else str(uuid4())
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="Trẻ em", ai_prefix="Bạn")
        user_sessions[user_id] = memory
        print(f"New session created with ID: {user_id}")

    agent = initialize_agent(
        tools,
        llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True,
        agent_kwargs={
            "system_message": SYSTEM_PROMPT,
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
        },
    )

    try:
        response = agent.invoke({"input": request_data})
        response_text = response["output"]
    except Exception as e:
        print(f"An error occurred while invoking the agent: {e}")
        return jsonify({"error": str(e)}), 500

    response_payload = {
        "id": user_id,
        "text": response_text
    }
    return jsonify(response_payload), 200

# Health check endpoint.
@app.route("/")
def read_root():
    return jsonify({"status": "OK", "message": "Agent service is running."})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("This server will listen for requests at http://127.0.0.1:8000")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8000)
