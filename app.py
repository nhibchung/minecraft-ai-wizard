import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

# Load local environment variables if testing on your computer
load_dotenv()

app = FastAPI(title="Minecraft AI Wizard")

# Fixes Unity WebGL cross-origin (CORS) errors so your browser can connect safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
INDEX_PATH = "faiss_minecraft_index"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable is missing!")

# 1. Initialize LangChain Serverless Embeddings API
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Initialize LangChain Serverless Text Generation API (LLM)
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.3, "max_new_tokens": 120}
)

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
    
    # Connect everything into a seamless Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    print("LangChain NPC System Active and Connected to Render!")

class ChatQuery(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "online", "message": "Render API is online. Send POST to /chat"}

@app.post("/chat")
async def chat(query: ChatQuery):
    if not qa_chain:
        return {"answer": "The NPC server is still initializing, please wait a moment."}
    try:
        response = qa_chain.run(query.question)
        return {"answer": response.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
