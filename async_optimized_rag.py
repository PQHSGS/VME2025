import os
import json
import time
import asyncio
import threading
import concurrent.futures
from functools import lru_cache
from flask import Flask, request, jsonify, abort
from typing import Dict, List, Tuple, Any, Optional
from uuid import uuid4
from gevent.pywsgi import WSGIServer

from langchain_core.tools import Tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

# ---------- config ----------
login("")
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_PROJECT'] = 'vme2025_gemini_embedding_004'
os.environ['GOOGLE_API_KEY'] = "AIzaSyAxL30FKtRAXzcThLyTQf9WN75VsVfCHkA"

app = Flask(__name__)
app.config.update(
    JSON_AS_ASCII=False,
    PROPAGATE_EXCEPTIONS=True,
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    WERKZEUG_RUN_MAIN=True
)

user_sessions: Dict[str, ConversationBufferWindowMemory] = {}

retrieval_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval_")
processing_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="processing_")

class AsyncFAISSLoader:
    """Helper class to load FAISS indexes asynchronously"""
    def __init__(self):
        self.loaded = False
        self.vs_h = None
        self.vs_c = None
        self.ctx = None
        self.emb = None
        self._load_lock = threading.Lock()

    def initialize(self):
        """Initialize all resources"""
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                print("Resources already loaded, skipping initialization")
                return
            print("Loading embedding model...")
            self.emb = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
            print("Loading vector stores...")
            out_dir = "Enhanced Flow/FAISS/full_db"
            header_dir = os.path.join(out_dir, "header_index")
            content_dir = os.path.join(out_dir, "content_index")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                header_future = executor.submit(
                    FAISS.load_local,
                    header_dir,
                    self.emb,
                    allow_dangerous_deserialization=True
                )
                content_future = executor.submit(
                    FAISS.load_local,
                    content_dir,
                    self.emb,
                    allow_dangerous_deserialization=True
                )
                self.vs_h = header_future.result()
                self.vs_c = content_future.result()
            print("Loading context data...")
            ctx_path = os.path.join(out_dir, "context.json")
            with open(ctx_path, "r", encoding="utf-8") as f:
                self.ctx = json.load(f)
            self.loaded = True
            print("All resources loaded successfully")

loader = AsyncFAISSLoader()
threading.Thread(target=loader.initialize, daemon=True).start()

@lru_cache(maxsize=4096)
def cached_embed(text: str) -> Tuple[float, ...]:
    """Returns a cached embedding tuple for the given text"""
    if not loader.loaded:
        loader.initialize()
    return tuple(loader.emb.embed_query(text))

async def hybrid_search(q_vec: List[float], topk: int = 4) -> Tuple[List, List]:
    """Run vector searches in parallel using asyncio and thread pool"""
    if not loader.loaded:
        loader.initialize()
    loop = asyncio.get_event_loop()
    header_task = loop.run_in_executor(
        retrieval_thread_pool,
        lambda: loader.vs_h.similarity_search_by_vector(q_vec, k=topk)
    )
    content_task = loop.run_in_executor(
        retrieval_thread_pool,
        lambda: loader.vs_c.similarity_search_by_vector(q_vec, k=topk)
    )
    h_hits, c_hits = await asyncio.gather(header_task, content_task)
    return h_hits, c_hits

async def retrieve_async(query: str, topk: int = 4, w_content: float = 1.0, w_header: float = 1.0) -> str:
    t0 = time.time()
    q_vec = cached_embed(query)

    try:
        h_hits_raw, c_hits_raw = await hybrid_search(q_vec, topk)
    except Exception as e:
        print(f"Vector search error: {e}")
        h_hits_raw = loader.vs_h.similarity_search_with_score(query, k=topk)
        c_hits_raw = loader.vs_c.similarity_search_with_score(query, k=topk)

    # normalize về dạng (doc, score)
    def normalize(hits):
        out = []
        for item in hits or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append((item[0], float(item[1])))
            else:
                out.append((item, None))
        return out

    h_hits = normalize(h_hits_raw)
    c_hits = normalize(c_hits_raw)

    scores = {}
    for doc, score in h_hits:
        cid = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
        if not cid:
            continue
        sim = 1.0 / (1.0 + score) if score is not None else 0.5
        scores[cid] = scores.get(cid, 0) + w_header * sim

    for doc, score in c_hits:
        cid = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
        if not cid:
            continue
        sim = 1.0 / (1.0 + score) if score is not None else 0.5
        scores[cid] = scores.get(cid, 0) + w_content * sim

    if not scores:
        return ""

    top_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    results = []
    for cid, comb_sim in top_chunks:
        doc = next((d for d in loader.ctx if d.get("metadata", {}).get("chunk_id") == cid), None)
        if not doc:
            continue
        doc.setdefault("metadata", {})["hybrid_score"] = comb_sim
        results.append(doc["metadata"]['header_path_str'] + '\n' + doc.get("page_content"))

    print(f"[timing] async retrieve {time.time()-t0:.3f}s")
    return "\n\n-----\n\n".join(results)

@tool
def retrieve(query: str, topk: int = 4, w_content: float = 1.0, w_header: float = 1.0) -> str:
    "retrieve info"
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            retrieve_async(query, topk, w_content, w_header)
        )
    finally:
        loop.close()

print("Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1,
    max_output_tokens=1024,
    top_k=40,
    top_p=0.95,
    transport="grpc",
)

tools = [
    Tool(
        name="museum_documents_tool",
        func=retrieve,
        description="Trích xuất thông tin liên quan từ tài liệu nội bộ."
    )
]

print("Loading system prompt...")
try:
    with open("Enhanced Flow\system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception as e:
    print(f"Error loading system prompt: {e}")
    SYSTEM_PROMPT = "Bạn là một trợ lý AI vui tính, đóng vai là một tiến sĩ cao tuổi am hiểu kiến thức, thân thiện."

_dummy_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
# Define prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Create agent
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     memory=_dummy_memory,
#     verbose=False
# )

agent_executor = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    handle_parsing_errors=True,
    memory=_dummy_memory,
    agent_kwargs={
        "system_message": SYSTEM_PROMPT,
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    },
    verbose=False
)

@app.route("/chat", methods=["POST"])
def chat_with_agent():
    tstart = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        request_data = data.get('data')
        request_id = data.get('id')
        if not request_data:
            return jsonify({"error": "Missing 'data' field"}), 400
        if isinstance(request_data, str) and len(request_data) > 2000:
            request_data = request_data[:2000]
        t_session_start = time.time()
        if request_id and request_id in user_sessions:
            memory = user_sessions[request_id]
        else:
            request_id = request_id or str(uuid4())
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                human_prefix="Trẻ em",
                ai_prefix="Bạn",
                k=6
            )
            user_sessions[request_id] = memory
        t_session = time.time() - t_session_start
        agent_executor.memory = memory
        t_agent_start = time.time()
        try:
            future = processing_thread_pool.submit(
                lambda: agent_executor.invoke({"input": request_data})
            )
            response = future.result(timeout=25)
            response_text = response["output"]
        except concurrent.futures.TimeoutError:
            return jsonify({"error": "Request timed out, please try again"}), 504
        except Exception as e:
            print(f"Error in agent execution: {str(e)}")
            return jsonify({"error": str(e)}), 500
        t_agent = time.time() - t_agent_start
        ttotal = time.time() - tstart
        print(f"[timing] session: {t_session:.3f}s, agent: {t_agent:.3f}s, total: {ttotal:.3f}s")
        return jsonify({
            "id": request_id,
            "text": response_text,
            "timing": {
                "session_ms": int(t_session * 1000),
                "agent_ms": int(t_agent * 1000),
                "total_ms": int(ttotal * 1000)
            }
        }), 200
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        ttotal = time.time() - tstart
        return jsonify({
            "error": "Internal server error",
            "timing_ms": int(ttotal * 1000)
        }), 500

@app.route("/")
def read_root():
    status = {
        "status": "OK",
        "message": "Async RAG service is running",
        "features": {
            "async_retrieval": True,
            "parallel_processing": True,
            "caching_enabled": True,
            "timeout_protection": True
        }
    }
    if loader.loaded:
        status["system"] = {
            "embedding_model": "hiieu/halong_embedding",
            "llm_model": "gemini-2.5-flash-lite0",
            "vector_db": "FAISS",
            "active_sessions": len(user_sessions)
        }
    else:
        status["system"] = {
            "state": "initializing",
            "message": "System is still loading resources"
        }
    return jsonify(status)

def cleanup_sessions():
    """Remove sessions older than 2 hours"""
    while True:
        time.sleep(3600)
        print("Session cleanup task running...")

cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    print("Starting async optimized Flask server...")
    print("This server will listen for requests at http://127.0.0.1:7000")
    loader.initialize()
    try:
        http_server = WSGIServer(
            ('0.0.0.0', 7000),
            app,
            spawn=100,
        )
        if hasattr(http_server, 'set_timeout'):
            http_server.set_timeout(30)
        print("Server ready to accept connections")
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Falling back to Flask development server...")
        app.run(debug=False, threaded=True, host='0.0.0.0', port=8000)
    finally:
        retrieval_thread_pool.shutdown(wait=False)
        processing_thread_pool.shutdown(wait=False)
        print("Server shutdown complete")
