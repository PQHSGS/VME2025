import os
import json
import time
import logging
import threading
import concurrent.futures
from functools import lru_cache
from flask import Flask, request, jsonify
from typing import Dict, List, Tuple, Any, Optional
from uuid import uuid4
from gevent.pywsgi import WSGIServer

from langchain_core.tools import Tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

# ---------- config ----------
login("")
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = ''
os.environ['LANGSMITH_PROJECT'] = 'vme2025_gemini_embedding_004'
os.environ['GOOGLE_API_KEY'] = ""

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

# Only keep one thread pool for processing agent requests
processing_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="processing_")

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    # Configure file handler for tsg.log
    file_handler = logging.FileHandler("tsg.log", encoding='utf-8')
    file_handler.setLevel(level)
    
    # Configure console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

setup_logging(verbose=True)  # Enable debug logging
logger = logging.getLogger(__name__)
class FAISSLoader:
    """Helper class to load FAISS indexes"""
    def __init__(self):
        self.loaded = False
        self.vs_h = None
        self.vs_c = None
        self.vs_q = None
        self.ctx = None
        self.question = None
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
            out_dir = "FAISS\\full_db\\"
            header_dir = os.path.join(out_dir, "header_index")
            content_dir = os.path.join(out_dir, "content_index")
            question_dir = os.path.join(out_dir, "question_index")
            # Load vector stores sequentially
            print("Loading header index...")
            self.vs_h = FAISS.load_local(
                header_dir,
                self.emb,
                allow_dangerous_deserialization=True
            )
            
            print("Loading content index...")
            self.vs_c = FAISS.load_local(
                content_dir,
                self.emb,
                allow_dangerous_deserialization=True
            )
            
            self.vs_q = FAISS.load_local(
                question_dir,
                self.emb,
                allow_dangerous_deserialization=True
            )
            print("Loading context data...")
            ctx_path = os.path.join(out_dir, "context.json")
            with open(ctx_path, "r", encoding="utf-8") as f:
                self.ctx = json.load(f)
            q_path = os.path.join(out_dir, "questions.json")
            with open(q_path, "r", encoding="utf-8") as f:
                self.question = json.load(f)
            self.loaded = True
            print("All resources loaded successfully")

loader = FAISSLoader()
threading.Thread(target=loader.initialize, daemon=True).start()

def hybrid_search(q_vec: List[float], topk: int = 4) -> Tuple[List, List]:
    """Run vector searches for header and content sequentially"""
    logger.debug(f"Starting hybrid search with vector of length {len(q_vec)}, topk={topk}")
    if not loader.loaded:
        logger.warning("Loader not initialized, initializing now...")
        loader.initialize()
    # Search both indexes sequentially
    logger.debug("Searching header index by vector...")
    h_hits = loader.vs_h.similarity_search_by_vector(q_vec, k=topk)
    logger.debug(f"Header search returned {len(h_hits)} hits")
    
    logger.debug("Searching content index by vector...")
    c_hits = loader.vs_c.similarity_search_by_vector(q_vec, k=topk)
    logger.debug(f"Content search returned {len(c_hits)} hits")
    
    logger.debug("Hybrid search completed")
    return h_hits, c_hits


@tool
def retrieve(query: str, topk: int = 4, w_content: float = 1.0, w_header: float = 1.0) -> str:
    """Retrieve relevant documents based on query"""
    t0 = time.time()
    logger.info(f"Starting retrieval for query: '{query}' (topk={topk}, w_content={w_content}, w_header={w_header})")
    
    # Search header index
    logger.debug("Searching header index...")
    h_hits_raw = loader.vs_h.similarity_search_with_score(query, k=topk)
    logger.debug(f"Header search returned {len(h_hits_raw)} results")
    
    # Search content index
    logger.debug("Searching content index...")
    c_hits_raw = loader.vs_c.similarity_search_with_score(query, k=topk)
    logger.debug(f"Content search returned {len(c_hits_raw)} results")

    # normalize to (doc, score) format
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

    # Calculate combined scores
    logger.debug("Calculating combined scores...")
    scores = {}
    header_chunks_processed = 0
    for doc, score in h_hits:
        cid = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
        if not cid:
            continue
        sim = 1.0 / (1.0 + score) if score is not None else 0.5
        scores[cid] = scores.get(cid, 0) + w_header * sim
        header_chunks_processed += 1

    content_chunks_processed = 0
    for doc, score in c_hits:
        cid = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
        if not cid:
            continue
        sim = 1.0 / (1.0 + score) if score is not None else 0.5
        scores[cid] = scores.get(cid, 0) + w_content * sim
        content_chunks_processed += 1
    logger.debug(f"Total unique chunks with scores: {len(scores)}")

    if not scores:
        logger.warning("No valid documents found with chunk_ids for the query")
        return ""

    # Sort by score and get top documents
    top_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    logger.debug(f"Selected top {len(top_chunks)} chunks based on hybrid scores")
    
    # Log the top chunk scores
    for i, (cid, score) in enumerate(top_chunks):
        logger.debug(f"  Rank {i+1}: chunk_id={cid}, hybrid_score={score:.4f}")

    # Format results
    results = []
    chunks_found = 0
    chunks_missing = 0
    for cid, comb_sim in top_chunks:
        doc = next((d for d in loader.ctx if d.get("metadata", {}).get("chunk_id") == cid), None)
        if not doc:
            chunks_missing += 1
            logger.warning(f"Could not find document for chunk_id: {cid}")
            continue
        chunks_found += 1
        doc.setdefault("metadata", {})["hybrid_score"] = comb_sim
        header_path = doc["metadata"].get('header_path_str', 'Unknown path')
        content_preview = doc.get("page_content", "")[:100] + "..." if len(doc.get("page_content", "")) > 100 else doc.get("page_content", "")
        logger.debug(f"Retrieved chunk {cid}: {header_path} | Content preview: {content_preview}")
        results.append(doc["metadata"]['header_path_str'] + '\n' + doc.get("page_content"))
    
    logger.info(f"Retrieval completed: {chunks_found} documents found, {chunks_missing} chunks missing")
    logger.debug(f"Retrieved {len(results)} documents for query: {query}")
    logger.debug(f"[timing] retrieve {time.time()-t0:.3f}s")
    return "\n\n-----\n\n".join(results)

def situation_match(query: str, thres: int = 1) -> List:
    """Find similar questions from the question index"""
    if not loader.loaded:
        loader.initialize()
    hits_raw = loader.vs_q.similarity_search_with_score(query, k=1)
    for item in hits_raw or []:
        doc, score = item[0], float(item[1])
        if score > thres:
            return None
        doc.metadata['score'] = score
    return doc

def preprocess_query(query: str) -> str:
    situation = situation_match(query)
    if situation:
        print('Special situation')
        question = situation.metadata.get('question', '')
        guidance = situation.metadata.get('guidance', '')
        answer = situation.metadata.get('answer', '')
        query = f"""\n\n---\n\n ###Trong lượt trả lời này, câu hỏi người dùng đã kích hoạt tình huống đặc biệt. Hãy tham khảo và đưa ra trả lời (phải chính xác theo văn phong câu trả lời mẫu) nếu tình huống gợi ý là phù hợp. Nếu không, hãy trả lời một cách tự nhiên và thân thiện như bình thường.
            *Tin nhắn người dùng: {query}
            *Tình huống gợi ý: {question}
            *Hướng dẫn trả lời: {guidance}
        """
        if answer != '':
            query += f"*Câu trả lời mẫu: {answer}"
    return query
    

print("Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
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
    with open("D:\Code\VME\Enhanced Flow\system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
        
except Exception as e:
    print(f"Error loading system prompt: {e}")
    SYSTEM_PROMPT = "Bạn là một trợ lý AI vui tính, đóng vai là một 'Tiến sĩ giấy' cao tuổi am hiểu kiến thức, thân thiện."

_dummy_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

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
    logger.info("Received new chat request")
    try:
        #receive data POST
        data = request.get_json()
        if not data:
            logger.warning("Invalid JSON received in request")
            return jsonify({"error": "Invalid JSON"}), 400
        request_data = data.get('data')
        request_id = data.get('id')
        if not request_data:
            logger.warning("Missing 'data' field in request")
            return jsonify({"error": "Missing 'data' field"}), 400
        if isinstance(request_data, str) and len(request_data) > 2000:
            logger.info(f"Truncating input from {len(request_data)} to 2000 characters")
            request_data = request_data[:2000]
        
        logger.debug(f"Request ID: {request_id}, Data length: {len(str(request_data))}")
        t_session_start = time.time()
        #Check for exist and create/match memory session
        if request_id and request_id in user_sessions:
            memory = user_sessions[request_id]
            logger.debug(f"Using existing session for ID: {request_id}")
        else:
            logger.debug(f"Creating new session with ID: {request_id if request_id else 'auto-generated'}")
            request_id = request_id or str(uuid4())
            logger.debug(f"Using request ID: {request_id}")
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                human_prefix="Trẻ em",
                ai_prefix="Ông tiến sĩ",
                k=6
            )
            user_sessions[request_id] = memory
            logger.debug(f"Created new session for ID: {request_id}, total sessions: {len(user_sessions)}")
        t_session = time.time() - t_session_start
        agent_executor.memory = memory
        t_agent_start = time.time()
        try:
            #input = preprocess_query(request_data)
            input = request_data
            print(input)
            logger.info(f"Processing user input: '{input}' for session {request_id}")
            future = processing_thread_pool.submit(
                lambda: agent_executor.invoke({"input": input})
            )
            response = future.result(timeout=25)
            response_text = response["output"]
            logger.info(f"Agent execution completed successfully for session {request_id}")
        except concurrent.futures.TimeoutError:
            logger.error(f"Agent execution timeout for session {request_id} after 25 seconds")
            return jsonify({"error": "Request timed out, please try again"}), 504
        except Exception as e:
            logger.error(f"Error in agent execution for session {request_id}: {str(e)}")
            print(f"Error in agent execution: {str(e)}")
            return jsonify({"error": str(e)}), 500
        t_agent = time.time() - t_agent_start
        ttotal = time.time() - tstart
        logger.info(f"Request completed successfully - Session: {t_session:.3f}s, Agent: {t_agent:.3f}s, Total: {ttotal:.3f}s")
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
        ttotal = time.time() - tstart
        logger.error(f"Unhandled error in chat_with_agent: {str(e)}", exc_info=True)
        print(f"Unhandled error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "timing_ms": int(ttotal * 1000)
        }), 500

@app.route("/")
def read_root():
    status = {
        "status": "OK",
        "message": "Simplified RAG service is running",
        "features": {
            "parallel_processing": False,
            "caching_enabled": True,
            "timeout_protection": True
        }
    }
    if loader.loaded:
        status["system"] = {
            "embedding_model": "hiieu/halong_embedding",
            "llm_model": "gemini-2.5-flash-lite",
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
    print("Starting simplified Flask server...")
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
        processing_thread_pool.shutdown(wait=False)
        print("Server shutdown complete")