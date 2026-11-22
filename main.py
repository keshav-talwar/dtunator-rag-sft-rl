# RAG model 

# Plan -> Make it so It fliters the data which is relevant to the student
# so 1st year student only get the data which is relevant to them and so on

import os 
import sys
import time
import logging
import re
import io
from pathlib import Path
from dataclasses import dataclass # --> Metadata for the chunks
from dotenv import load_dotenv
from PIL import Image 
import pytesseract 
import pymupdf #to scan the pdf file and extract the text 
from pymupdf import Page
import chromadb #vector Database 
from chromadb.config import Settings 
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse # 
import requests
from sklearn import metrics # Evaluation Metrics
from sklearn.metrics import precision_recall_fscore_support # Evaluation Metrics
from typing import List, Dict

# Fine Tuning and evaluation 
# Reinforcement Learning ( Model -> Llama , Reward Model -> Human Feeback and Chatgpt)

app = FastAPI(title="DTUnator RAG API")

load_dotenv()
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)



#Vector Database 
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "collection"



MAX_HISTORY_TURNS = 5
WHISPER_MODEL_SIZE = "distil-medium.en"

# Reward Model for RLHF
ONDEMAND_API_KEY = os.getenv("ONDEMAND_API_KEY")
ONDEMAND_URL = "https://api.on-demand.io/chat/v1/sessions/query" 
ONDEMAND_ENDPOINT_ID = "predefined-openai-gpt4.1-nano"




logger.info("Loading tokenizer...")
#  BM3
#  NV embed
tokenizer = AutoTokenizer.from_pretrained('''"sentence-transformers/all-MiniLM-L6-v2"Place holder''') #Tokenizing the text data to store it in chromaDB and used for CrossEncoder 

logger.info("Loading cross encoder...")


cross_encoder = CrossEncoder('''"cross-encoder/ms-marco-MiniLM-L-6-v2" Place holder''') #Reranking Model - Improve the revelance of the docs

#No clue how to use 


chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DB_PATH,
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def clean_text(text): #It removes noise 
    text = re.sub(r"\d+/\d+", "", text) 
    text = re.sub(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", "", text) 
    text = re.sub(r"[^\x00-\x7F]+", " ", text) 
    text = re.sub(r"\n{3,}", "\n\n", text) 
    return text.strip()



def chunks_sematically(text,max_tokens=200)-> List[str]:
    embeddings = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    total_tokens = len(embeddings)
    chunks = []
    start = 0
    while start < total_tokens:
        if start + max_tokens >= total_tokens:
            chunk = tokenizer.decode(embeddings[start:total_tokens], skip_special_tokens=False)
            chunks.append(chunk)
            break
        else: # Try to find a natural break point (newline) within the next max_tokens
            chunk = tokenizer.decode(embeddings[start:start+max_tokens], skip_special_tokens=False)
            last_newline = chunk.rfind("\n")
            if last_newline > max_tokens // 2:
                chunk = chunk[:last_newline]
                start += len(tokenizer(chunk, return_tensors="pt", truncation=False)["input_ids"][0])
            else:
                start += max_tokens
            chunks.append(chunk)
    return chunks




def add_chunks_to_chroma(chunks): # Each chunk is a dict with "id", "text->actual data ", " source -> metadata "
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"]})

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

def process_scanned_page(page): # 
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return pytesseract.image_to_string(image, lang="eng")


def user_sends_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    ext = file_path.suffix.lower()
    all_text = ""

    if ext == ".pdf":
        doc = pymupdf.open(file_path)
        for page in doc:
            raw_text = page.get_text()

            if isinstance(raw_text, str):
                text = raw_text.strip()
            else:
                text = ""

            if len(text) < 50:
                all_text += process_scanned_page(page)
            else:
                all_text += text
        doc.close()

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            all_text = f.read()

    else:
        return []

    cleaned = clean_text(all_text)
    chunks = chunks_sematically(cleaned, max_tokens=200)
    add_chunks_to_chroma(chunks)

    return chunks



def retrieve_docs_from_chroma(query, k=10):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = []
    if results and results.get("documents") and results["documents"] and results.get("metadatas") and results["metadatas"]:
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

    return docs


def cross_rerank(query, docs, top_k=5):
    if not docs:
        return []

    pairs = [(query, d["content"]) for d in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    results = []
    for i, (doc, score) in enumerate(ranked[:top_k]):
        results.append({
            "rank": i + 1,
            "score": float(score),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    return results

def format_history(messages):
    history = messages[-MAX_HISTORY_TURNS * 2 :]
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm_api(prompt, system_instruction):
    payload = {
        "endpointId": ONDEMAND_ENDPOINT_ID,
        "responseMode": "sync",
        "query": f"{system_instruction}\n\n{prompt}",
    }

    headers = {
        "apikey": ONDEMAND_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(ONDEMAND_URL, json=payload, headers=headers, timeout=60)
    return response.json()["data"]["answer"]


def generate_rag_response(user_query, history):
    retrieved = retrieve_docs_from_chroma(user_query)
    reranked = cross_rerank(user_query, retrieved)

    # HARD HALLUCINATION GATE
    if not reranked or reranked[0]["score"] < 4.5:
        return "I do not have enough verified DTU information to answer this." # also give result to the reward `model` for RLHF
    

    history_block = format_history(history)
    context_text = "\n\n".join(d["content"] for d in reranked)

    prompt = f"""
Conversation:
{history_block}

Context:
{context_text}

Question:
{user_query}
"""

    system_instruction = (
        "You are DTUnator.\n"
        "Answer strictly from provided Context.\n"
        "If answer is not fully supported, say you do not know.\n"
        "Do NOT use prior knowledge.\n"
        "Be accurate and concise."
    )

    return call_llm_api(prompt, system_instruction)


class LLMService:
    def generate_response(self, question: str, history: list):
        response = generate_rag_response(question, history)
        for word in response.split():
            yield word + " "
            time.sleep(0.01)

llm_service = LLMService()


@app.post("/chat")
def chat_endpoint(payload: dict):
    question = payload.get("question")
    history = payload.get("history", [])

    if not question or not isinstance(question, str):
        return {"error": "question field is required and must be a string"}

    return StreamingResponse(
        llm_service.generate_response(question, history),
        media_type="text/plain"
    )
