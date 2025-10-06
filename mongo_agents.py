from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timezone
import os, bcrypt
from pymongo.errors import ConnectionFailure

# Import Groq agent functions
from agents import route_and_answer, generate_chat_title, summarizer_llm  

# === Connect to MongoDB ===
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

DB_NAME = "CapstoneDB"
if DB_NAME not in client.list_database_names():
    print(f"Creating new database: {DB_NAME}")
else:
    print(f"Using existing database: {DB_NAME}")

agents_db = client[DB_NAME]

# Collections
users_col = agents_db["users"]
chats_col = agents_db["chats"]
messages_col = agents_db["messages"]


# Password helpers
def hash_password(password: str) -> bytes:
    if isinstance(password, str):
        password = password.encode("utf-8")
    return bcrypt.hashpw(password, bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    if isinstance(password, str):
        password = password.encode("utf-8")
    return bcrypt.checkpw(password, hashed)

# User Management
def signup_user(username, password, email):
    if users_col.find_one({"username": username}):
        return " Username already exists"

    user = {
        "username": username,
        "password_hash": hash_password(password),
        "email": email,
        "created_at": datetime.now(timezone.utc)
    }
    users_col.insert_one(user)
    return " User created successfully"

def login_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return None
    if verify_password(password, user["password_hash"]):
        return user
    return None

# Chat Management
def create_new_chat(user_id, db=agents_db):
    chat = {
        "user_id": ObjectId(user_id),
        "title": "Untitled Chat",
        "created_at": datetime.now(timezone.utc)
    }
    result = db["chats"].insert_one(chat)
    return str(result.inserted_id)

def update_chat_title(chat_id, new_title, first_query=None, db=agents_db):
    update_fields = {"title": new_title[:50]}
    if first_query:
        update_fields["first_query"] = first_query
    db.chats.update_one({"_id": ObjectId(chat_id)}, {"$set": update_fields})

def save_message(chat_id, role, content, db=agents_db):
    message = {
        "chat_id": ObjectId(chat_id),
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    }
    db["messages"].insert_one(message)

def get_chat_history(chat_id, db=agents_db):
    msgs = list(db["messages"].find({"chat_id": ObjectId(chat_id)}).sort("timestamp", 1))
    return [(m["role"], m["content"]) for m in msgs]

def get_user_chats(user_id, db=agents_db):
    return list(db["chats"].find({"user_id": ObjectId(user_id)}).sort("created_at", -1))

def get_chat_summary(chat_id, db=agents_db):
    """Generate summary of conversation using summarizer_llm."""
    history = get_chat_history(chat_id, db)
    if not history:
        return ""
    convo_text = "\n".join([f"{r}: {c}" for r, c in history])
    return summarizer_llm(convo_text)


# Ask Questions (Agents + Save to DB)
def ask_question(user, chat_id, query, db):
    # Store user message
    save_message(chat_id, "user", query, db)

    # Set chat title if first query
    chat = db["chats"].find_one({"_id": ObjectId(chat_id)})
    msg_count = db["messages"].count_documents({"chat_id": ObjectId(chat_id)})
    if msg_count == 1 and chat.get("title", "Untitled Chat") == "Untitled Chat":
        ai_title = generate_chat_title(query)
        update_chat_title(chat_id, ai_title, first_query=query, db=db)

    # Get 4 outputs from agent pipeline
    answer, retrieved_contexts, tool_used, metrics = route_and_answer(user["_id"], query)

    # Store assistant message
    save_message(chat_id, "assistant", answer, db)

    # Return only answer + metrics
    return answer, metrics,tool_used

# Delete Chat + Messages
def delete_chat(chat_id: str, db=agents_db) -> str:
    try:
        obj_id = ObjectId(chat_id)
        db["messages"].delete_many({"chat_id": obj_id})
        result = db["chats"].delete_one({"_id": obj_id})
        if result.deleted_count > 0:
            return "Chat deleted successfully."
        else:
            return "Chat not found."
    except Exception as e:
        return f"Error deleting chat: {e}"
