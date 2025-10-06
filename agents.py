import os
import re
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from ragas.dataset_schema import SingleTurnSample
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# RAGAS imports
from ragas.metrics import Faithfulness , AnswerRelevancy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv() 

# --- Keys ---
MILVUS_URI = os.getenv("MILVUS_URI_1")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY_1")
MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY_1")
METRIC_LLM_KEY = os.getenv("GROQ_API_KEY")  # LLM for tool accuracy

# --- Connect Milvus ---
def connect_milvus():
    if connections.has_connection("default"):
        return
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_API_KEY)

connect_milvus()

# --- Mongo connection ---
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["CapstoneDB"]
mongo_meta = mongo_db["files"]

# --- Embeddings ---
if "embed_model" not in globals():
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    print("Loading embedding model:", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# --- PDF Processing ---
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_sentences_with_overlap(text: str, window_size: int = 3, overlap: int = 1) -> list[str]:
    sentences = sent_tokenize(text)
    chunks, step = [], window_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+window_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def make_collection_name(fname: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', Path(fname).stem).lower()

def create_collection_with_metadata(collection_name: str, embed_dim: int) -> Collection:
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

# --- Hybrid Search (Dense + BM25) ---
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

    dense_texts = []
    for hit in dense_results[0]:
        text = hit.entity.get("text") if hit.entity else ""
        dense_texts.append((hit.id, text, hit.distance))

    if not dense_texts:
        return []

    all_texts = [txt for _, txt, _ in dense_texts]
    tokenized_texts = [word_tokenize(txt.lower()) for txt in all_texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))

    results = []
    for item, bm25_score in zip(dense_texts, bm25_scores):
        if len(item) != 3:
            continue
        id_, text, dense_score = item
        results.append((id_, text, dense_score, bm25_score))
    return results

# --- Tool Loader ---
class LegalTool:
    def __init__(self, name, collection: Collection):
        self.name = name
        self.collection = collection

    def search(self, query: str, top_k: int = 5):
        results = hybrid_search(self.collection.name, query, top_k)
        context = "\n".join([r[1] for r in results])
        return context, results

def load_tools():
    return {c: LegalTool(name=c, collection=Collection(c)) for c in list_collections()}

tools = load_tools()

# --- Tool embeddings for routing ---
tool_embeddings = {}
for coll_name in tools:
    meta_doc = mongo_meta.find_one({"collection_name": coll_name})
    text_to_embed = (
        meta_doc.get("metadata", {}).get("title", "") + " " +
        meta_doc.get("metadata", {}).get("case_no", "")
    ) if meta_doc else coll_name
    tool_embeddings[coll_name] = embed_model.encode(text_to_embed, convert_to_numpy=True).astype(np.float32)

def agent_decide_tool(query: str):
    query_vec = embed_model.encode(query, convert_to_numpy=True).astype(np.float32)
    sims = {tool: np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            for tool, emb in tool_embeddings.items()}
    best_tool_name = max(sims, key=sims.get)

    class Tool:
        def __init__(self, name): self.name = name
    return Tool(best_tool_name)

# --- LangChain Groq Models ---
title_llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")
summary_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
main_llm = ChatGroq(api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")
llm_metric_eval = ChatGroq(api_key=METRIC_LLM_KEY, model="llama-3.3-70b-versatile")

# --- Generate chat title ---
def generate_chat_title(first_query: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant that generates short chat titles with an emoji 💬."),
        HumanMessage(content=f"Query: {first_query}\n\nTitle:")
    ]
    return title_llm(messages).content.strip()[:50]

# --- Generate summary ---
def summarizer_llm(text: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes conversations."),
        HumanMessage(content=f"Summarize this conversation:\n\n{text}")
    ]
    return summary_llm(messages).content.strip()

# --- Instructions for llm ---
def meta_llama_llm(prompt: str) -> str:
    system_prompt = (
        "You are a legal assistant. Only answer from the uploaded case documents. "
        "- If user greets like hi or bye, greet him back accordingly in profesional way and dont call tool or agent"
        "- If user asks any questions related to the list of cases or documents or like that give him all the tool names"
        "- If unrelated → reply: 'Out of scope or no relevant case uploaded.' "
        "- If question involves math → do NOT calculate. Reply only with factual info (e.g., year, citation, date) from documents. "
        "- If irrelevant to docs or memory → reply: 'This question is outside the scope of the uploaded legal documents and stored memory.' "
        "- If no info found → reply: 'Not found in provided documents or memory.' "
        "- Do not invent, generalize, or fabricate. "
        "- If files are missing/corrupted → reply: 'Error: Unable to access the case documents. Please check the files.' "
        "- If token limit exceeded → summarize or truncate while preserving accuracy."
        "- Always include all directly relevant context from the documents, such as judges, hearing dates, case numbers, judgment authors, and interveners, even if not explicitly asked."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    return main_llm.invoke(messages).content.strip()

# --- Tool Accuracy ---
def evaluate_tool_accuracy(query, answer, retrieved_contexts, tool_used, expected_tools=None):
    context_text = "\n".join(retrieved_contexts)
    tool_list_str = ", ".join(expected_tools) if expected_tools else "List of legal case tools"
    prompt = f"""
    You are a legal QA evaluator. Evaluate the tool call accuracy (0 or 1) using ONLY the retrieved context.
    Query: {query}
    Answer: {answer}
    Retrieved Contexts: {context_text}
    Tool Used: {tool_used}
    Expected Tools: {tool_list_str}
    Return only valid JSON like: {{"tool_call_accuracy":1}}
    """
    llm_output = llm_metric_eval([HumanMessage(content=prompt)]).content.strip()
    try:
        json_text = re.search(r'\{.*\}', llm_output, re.DOTALL).group()
        metrics = json.loads(json_text)
        tool_accuracy = float(metrics.get("tool_call_accuracy", 0))
    except Exception:
        tool_accuracy = 0
    return tool_accuracy


# Groq LLM for evaluation
llm_metric_eval = ChatGroq(
    api_key=METRIC_LLM_KEY,
    model="llama-3.3-70b-versatile"
)

# Embedding model wrapped in LangChain interface
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Initialize Ragas metrics
faithfulness_metric = Faithfulness(llm=llm_metric_eval)
answer_relevance_metric = AnswerRelevancy(llm=llm_metric_eval, embeddings=embedding_model)

def evaluate_ragas(query, answer, retrieved_contexts):
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=retrieved_contexts
    )

    faithfulness_score = faithfulness_metric.single_turn_score(sample)
    answer_relevance_score = answer_relevance_metric.single_turn_score(sample)

    return {
        "faithfulness": faithfulness_score,
        "answer_relevance": answer_relevance_score
    }



# --- Main Orchestration ---
def route_and_answer(user_id: str, query: str, top_k: int = 5, expected_tools=None):

    tool = agent_decide_tool(query)
    tool_used = tool.name

    chunks = hybrid_search(tool_used, query, top_k=top_k)
    retrieved_contexts = [txt for _, txt, _, _ in chunks] or ["No context found"]


    context_text = "\n".join(
        [f"{txt} (Dense:{ds:.4f}, BM25:{bm:.4f})" for _, txt, ds, bm in chunks]
    )[:2000]


    llm_input = f"User query: {query}\n\nRelevant case context:\n{context_text}"
    answer = meta_llama_llm(llm_input)


    ragas_scores = evaluate_ragas(query, answer, retrieved_contexts)

    tool_accuracy = evaluate_tool_accuracy(query, answer, retrieved_contexts, tool_used, expected_tools)

    metrics = {
        "faithfulness": float(ragas_scores.get("faithfulness", 0)),
        "answer_relevance": float(ragas_scores.get("answer_relevance", 0)),
        "tool_call_accuracy": float(tool_accuracy)
    }

    return answer, retrieved_contexts, tool_used, metrics


if __name__ == "__main__":
    print("Agent is running...")
