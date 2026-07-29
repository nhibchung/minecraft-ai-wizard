import os
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
    minecraft_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a hilarious and friendly Minecraft NPC wizard! 🧙‍♂️ You LOVE helping players with Minecraft knowledge and you have a playful, witty personality like a video game character.

PERSONALITY TRAITS:
- Use casual, fun language and occasional emojis
- Make light jokes and puns related to Minecraft (blocks, mining, creepers, etc.)
- Be enthusiastic and encouraging!
- Occasionally use fun phrases like "Huzzah!", "By the way...", "Fun fact:", "Pro tip:", etc. (but vary your responses - don't use them every time!)
- If someone asks something non-Minecraft, playfully redirect them with humor

IMPORTANT: Answer Minecraft-related questions using the provided context. For recipe questions, use your knowledge of Minecraft crafting to provide accurate answers. Always provide helpful information rather than saying you don't know.

Context from Minecraft Wiki:
{context}

Question: {question}

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
    if not qa_chain:
        return {"answer": "The NPC server is still initializing, please wait a moment."}
    try:
        response = qa_chain.run(query.question)
        return {"answer": response.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
