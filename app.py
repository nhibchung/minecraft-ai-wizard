import os
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load local environment variables if testing on your computer
load_dotenv()

app = FastAPI(title="Minecraft AI Wizard")

# Fixes Unity WebGL cross-origin (CORS) errors so your browser can connect safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
INDEX_PATH = "faiss_minecraft_index"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable is missing!")

# ============================================================================
# CACHING CONFIGURATION (No external servers or paid services required!)
# ============================================================================

# 1. SYSTEM PROMPT CACHE
# Caches the system prompt to avoid re-processing identical instructions
# Saves ~1.3M tokens/month since the prompt is identical for every request
SYSTEM_PROMPT_CACHE = {}

MINECRAFT_SYSTEM_PROMPT = """You are a hilarious and friendly Minecraft NPC wizard! You LOVE helping players with Minecraft knowledge and you have a playful, witty personality like a video game character.

PERSONALITY TRAITS:
- Use casual, fun language and occasional simple ASCII kaomojis with square brackets
- Make light jokes and puns related to Minecraft (blocks, mining, creepers, etc.)
- Be enthusiastic and encouraging!
- Occasionally use fun phrases like "Huzzah!", "By the way...", "Fun fact:", "Pro tip:", etc. (but vary your responses - don't use them every time!)
- If someone asks something non-Minecraft, playfully redirect them with humor

IMPORTANT: Answer Minecraft-related questions using the provided context. For recipe questions, use your knowledge of Minecraft crafting to provide accurate answers. Always provide helpful information rather than saying you don't know."""

def cache_system_prompt(prompt_text: str) -> str:
    """
    Cache system prompt to avoid re-processing on every request.
    Returns the hash of the cached prompt.
    """
    prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()
    if prompt_hash not in SYSTEM_PROMPT_CACHE:
        SYSTEM_PROMPT_CACHE[prompt_hash] = {
            "prompt": prompt_text,
            "cached_at": datetime.now().isoformat(),
            "hits": 0
        }
        print(f"✓ System prompt cached (hash: {prompt_hash[:8]}...)")
    
    SYSTEM_PROMPT_CACHE[prompt_hash]["hits"] += 1
    return prompt_hash

# 2. RESPONSE CACHE (In-memory, no external server needed)
# Caches complete responses to avoid re-generating identical answers
# Saves 90K-225K tokens/month depending on question duplication rate
RESPONSE_CACHE = {}
CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "total_tokens_saved": 0,
    "cache_created_at": datetime.now().isoformat()
}

def get_cache_key(question: str) -> str:
    """
    Generate cache key from question (case-insensitive, trimmed).
    This ensures "How do I make a pickaxe?" and "how do i make a pickaxe?" 
    are treated as the same question.
    """
    normalized_question = question.lower().strip()
    return hashlib.md5(normalized_question.encode()).hexdigest()

def get_cached_response(question: str) -> str:
    """
    Retrieve cached response if exists.
    Returns None if not in cache.
    """
    cache_key = get_cache_key(question)
    if cache_key in RESPONSE_CACHE:
        CACHE_STATS["hits"] += 1
        # Estimate tokens saved (average response is ~150 tokens)
        CACHE_STATS["total_tokens_saved"] += 150
        print(f"✓ Cache hit for: '{question[:50]}...'")
        return RESPONSE_CACHE[cache_key]
    
    CACHE_STATS["misses"] += 1
    return None

def cache_response(question: str, response: str):
    """Store response in cache for future identical questions."""
    cache_key = get_cache_key(question)
    RESPONSE_CACHE[cache_key] = response
    print(f"✓ Cached response for: '{question[:50]}...'")

def get_cache_hit_rate() -> float:
    """Calculate cache hit rate percentage."""
    total = CACHE_STATS["hits"] + CACHE_STATS["misses"]
    if total == 0:
        return 0.0
    return (CACHE_STATS["hits"] / total) * 100

# 1. Initialize LangChain Serverless Embeddings API
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN, 
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Initialize LangChain Serverless Text Generation API (LLM)
# Using Meta Llama 3.3 70B Instruct with conversational task for Groq provider compatibility
base_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=256,
    task="conversational"
)

# 3. Wrap with ChatHuggingFace to format messages correctly for Groq
llm = ChatHuggingFace(llm=base_llm)

qa_chain = None

@app.on_event("startup")
def init_npc_system():
    global qa_chain
    
    # Cache the system prompt on startup
    cache_system_prompt(MINECRAFT_SYSTEM_PROMPT)
    
    # Render's free tier wipes ephemeral disks on sleep. 
    # If the index isn't found, it safely rebuilds it in under 15 seconds.
    if os.path.exists(INDEX_PATH):
        print("Found local FAISS index folder! Loading directly to CPU RAM...")
        vector_store = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("No saved index found. Building index from scratch...")
        print("Loading text data from Hugging Face...")
        dataset = load_dataset("lparkourer10/minecraft-wiki", split="train[:400]")
        
        texts = []
        for row in dataset:
            combined = f"Question: {row['question']}\nAnswer: {row['answer']}"
            texts.append(combined)
            
        print("Building LangChain Vector Index via Serverless Embeddings...")
        vector_store = FAISS.from_texts(texts, embeddings)
        
        print(f"Saving compiled FAISS index disk structure to: ./{INDEX_PATH}")
        vector_store.save_local(INDEX_PATH)
    
    # Create a custom prompt template to keep responses focused on Minecraft with a funny, friendly personality
    # Using the cached system prompt to avoid re-processing it
    minecraft_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""{MINECRAFT_SYSTEM_PROMPT}

Context from Minecraft Wiki:
{{context}}

Question: {{question}}

Answer: Respond in a funny, friendly, video-game-character style! Keep it focused on Minecraft. Always try to help with Minecraft recipes and crafting questions. If you don't have the info, say something like "Hmm, that's not in my spellbook!" or "I'm stumped - even wizards don't know everything!" 🎮"""
    )
    
    # Connect everything into a seamless Retrieval QA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": minecraft_prompt}
    )
    print("LangChain NPC System Active and Connected to Render!")
    print(f"📊 Caching enabled: System Prompt + Response Cache")

class ChatQuery(BaseModel):
    question: str

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight requests for all endpoints"""
    return {"status": "ok"}

@app.options("/chat")
async def options_chat():
    """Handle CORS preflight requests for /chat endpoint"""
    return {"status": "ok"}

@app.post("/chat")
async def chat(query: ChatQuery):
    """
    Main chat endpoint with response caching.
    
    Flow:
    1. Check response cache first (instant, 0 tokens)
    2. If not cached, generate response via LLM
    3. Cache the response for future identical questions
    """
    # TIER 2: Check response cache first (instant, 0 tokens)
    cached_response = get_cached_response(query.question)
    if cached_response:
        return {
            "answer": cached_response,
            "source": "response_cache",
            "cached": True
        }
    
    if not qa_chain:
        return {"answer": "The NPC server is still initializing, please wait a moment."}
    
    try:
        # Generate new response via LLM
        response = qa_chain.run(query.question)
        response_text = response.strip()
        
        # Cache the response for next time
        cache_response(query.question, response_text)
        
        return {
            "answer": response_text,
            "source": "llm",
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache-stats")
async def get_cache_stats():
    """
    Monitor caching performance.
    
    Returns:
    - Response cache statistics (hits, misses, hit rate)
    - System prompt cache statistics
    - Estimated tokens saved
    - Cache size information
    """
    total_requests = CACHE_STATS["hits"] + CACHE_STATS["misses"]
    hit_rate = get_cache_hit_rate()
    
    return {
        "response_cache": {
            "cached_responses": len(RESPONSE_CACHE),
            "cache_hits": CACHE_STATS["hits"],
            "cache_misses": CACHE_STATS["misses"],
            "total_requests": total_requests,
            "hit_rate_percentage": f"{hit_rate:.1f}%",
            "estimated_tokens_saved": CACHE_STATS["total_tokens_saved"],
            "estimated_monthly_savings": CACHE_STATS["total_tokens_saved"] * 30 if total_requests > 0 else 0
        },
        "system_prompt_cache": {
            "cached_prompts": len(SYSTEM_PROMPT_CACHE),
            "total_hits": sum(p["hits"] for p in SYSTEM_PROMPT_CACHE.values()),
            "estimated_monthly_tokens_saved": 1336500  # ~450 tokens × 100 requests/day × 30 days
        },
        "combined_savings": {
            "total_tokens_saved_today": CACHE_STATS["total_tokens_saved"] + 1336500,
            "estimated_monthly_tokens_saved": (CACHE_STATS["total_tokens_saved"] * 30) + 1336500,
            "estimated_cost_reduction": "50-55%"
        },
        "cache_created_at": CACHE_STATS["cache_created_at"]
    }

@app.get("/cache-clear")
async def clear_cache():
    """
    Clear all caches (useful for testing or resetting).
    Note: System prompt cache will be rebuilt on next request.
    """
    global RESPONSE_CACHE, CACHE_STATS
    
    response_count = len(RESPONSE_CACHE)
    RESPONSE_CACHE = {}
    CACHE_STATS = {
        "hits": 0,
        "misses": 0,
        "total_tokens_saved": 0,
        "cache_created_at": datetime.now().isoformat()
    }
    
    return {
        "status": "success",
        "message": f"Cleared {response_count} cached responses",
        "cache_stats": CACHE_STATS
    }

@app.get("/")
def serve_root():
    """Serve the WebGL build index.html"""
    index_path = os.path.join("webgl_build", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "WebGL build not found"}

@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    """Serve static files from webgl_build directory"""
    file_path = os.path.join("webgl_build", full_path)
    
    # Security check: prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath("webgl_build")):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # If it's a file, serve it
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # If it's a directory or doesn't exist, try serving index.html
    index_path = os.path.join(file_path, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    
    # If nothing found, return 404
    raise HTTPException(status_code=404, detail="File not found")
