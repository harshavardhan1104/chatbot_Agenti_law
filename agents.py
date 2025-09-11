# agents.py (LangChain Orchestration with 3 Groq Models)

import os
import re
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# NLP / Embedding
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Milvus & Mongo
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from pymongo import MongoClient

# LangChain imports
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load env
load_dotenv()

# --- Keys
MILVUS_URI = os.getenv("MILVUS_URI_1")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY_1")
MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY_1")

# --- Connect Milvus
def connect_milvus():
    if connections.has_connection("default"):
        return
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_API_KEY)

connect_milvus()

# --- Mongo connection
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["CapstoneDB"]
mongo_meta = mongo_db["files"]

# --- Embeddings
if "embed_model" not in globals():
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    print("Loading embedding model:", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# --- PDF Processing
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_sentences_with_overlap(text, window_size=3, overlap=1):
    sentences = sent_tokenize(text)
    chunks, step = [], window_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+window_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def make_collection_name(fname: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', Path(fname).stem).lower()

def create_collection_with_metadata(collection_name, embed_dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
    ]
    schema = CollectionSchema(fields, description="Legal case chunks with metadata")
    coll = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}}
    coll.create_index(field_name="dense_vector", index_params=index_params)
    coll.load()
    return coll

# --- Hybrid Search (Dense + BM25)
def hybrid_search(collection_name: str, query: str, top_k: int = 5):
    collection = Collection(collection_name)
    collection.load()
    query_vec = embed_model.encode([query])[0].tolist()
    dense_results = collection.search(
        data=[query_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id", "text"]
    )
    dense_texts = [(hit.id, hit.entity.get("text"), hit.distance) for hit in dense_results[0]]
    if not dense_texts:
        return []
    all_texts = [t[1] for t in dense_texts if t[1]]
    tokenized_texts = [word_tokenize(txt.lower()) for txt in all_texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
    return [(id_, text, dense_score, bm25_score) for (id_, text, dense_score), bm25_score in zip(dense_texts, bm25_scores)]

# --- Tool Loader
class LegalTool:
    def __init__(self, name, collection):
        self.name = name
        self.collection = collection
    def search(self, query: str, top_k: int = 5):
        results = hybrid_search(self.collection, query, top_k)
        context = "\n".join([r[1] for r in results])
        return context, results

def load_tools():
    return {c: LegalTool(name=c, collection=Collection(c)) for c in list_collections()}

tools = load_tools()

# --- Tool embeddings for routing
tool_embeddings = {}
for coll_name in tools:
    meta_doc = mongo_meta.find_one({"collection_name": coll_name})
    text_to_embed = (meta_doc.get("metadata", {}).get("title", "") + " " + meta_doc.get("metadata", {}).get("case_no", "")) if meta_doc else coll_name
    tool_embeddings[coll_name] = embed_model.encode(text_to_embed, convert_to_numpy=True).astype(np.float32)

def agent_decide_tool(query: str):
    query_vec = embed_model.encode(query, convert_to_numpy=True).astype(np.float32)
    sims = {tool: np.dot(query_vec, emb) / (np.linalg.norm(query_vec)*np.linalg.norm(emb)) for tool, emb in tool_embeddings.items()}
    best_tool_name = max(sims, key=sims.get)
    class Tool:
        def __init__(self, name): self.name = name
    return Tool(best_tool_name)

# --- LangChain Groq Models
title_llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")
summary_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
main_llm = ChatGroq(api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")

def generate_chat_title(first_query: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant that generates short chat titles with an emoji 💬."),
        HumanMessage(content=f"Query: {first_query}\n\nTitle:")
    ]
    return title_llm(messages).content.strip()[:50]

def summarizer_llm(text: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes conversations."),
        HumanMessage(content=f"Summarize this conversation:\n\n{text}")
    ]
    return summary_llm(messages).content.strip()

def meta_llama_llm(prompt: str) -> str:
    messages = [
        SystemMessage(content="You are a legal assistant. Only answer from provided case context. If unrelated, say 'Out of scope or no relevant case uploaded.'"),
        HumanMessage(content=prompt)
    ]
    return main_llm(messages).content.strip()

# --- Main Orchestration
def route_and_answer(user_id: str, query: str, top_k: int = 5):
    tool = agent_decide_tool(query)
    tool_used = tool.name
    chunks = hybrid_search(tool_used, query, top_k=top_k)
    context_text = "\n".join([f"{txt} (Dense:{ds:.4f}, BM25:{bm:.4f})" for _, txt, ds, bm in chunks])
    llm_input = f"User query: {query}\n\nRelevant case context:\n{context_text}"
    answer = meta_llama_llm(llm_input)
    # ⚠️ Messages are not saved here — handled only in mongo_agents.py
    return answer, chunks, tool_used


if __name__ == "__main__":
    print("Agents (LangChain + 3 Groq models) loaded successfully!")
