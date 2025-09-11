import streamlit as st
import os
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient
from functools import lru_cache
import time
from mongo_agents import ask_question

# Import agent functions
from agents import route_and_answer, load_tools
from mongo_agents import (
    signup_user, login_user, create_new_chat, get_user_chats,
    get_chat_history, get_chat_summary
)

# --- Page Config ---
st.set_page_config(
    page_title="Legal Case Bot",
    page_icon="⚖️",
    layout="wide",
)

# --- Initialize session state early ---
def init_session_state():
    """Initialize session state variables once"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tools_loaded" not in st.session_state:
        st.session_state.tools_loaded = False
    if "user" not in st.session_state:
        st.session_state.user = None

# --- Cached MongoDB Connection ---
@st.cache_resource
def get_mongo_client():
    """Create MongoDB connection once and cache it"""
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        st.error("MONGO_URI environment variable not set")
        st.stop()
    return MongoClient(MONGO_URI)

@st.cache_resource
def get_agents_db():
    """Get database connection once and cache it"""
    mongo_client = get_mongo_client()
    return mongo_client["CapstoneDB"]

# --- Cached tools loading ---
@st.cache_resource
def get_tools():
    """Load tools once and cache them"""
    return load_tools()

# --- Cached CSS loading ---
@st.cache_data
def load_css():
    """Load CSS once and cache it"""
    try:
        with open("style.css") as f:
            return f.read()
    except FileNotFoundError:
        st.warning("style.css file not found. Using default styling.")
        return ""

# --- Connection check ---
@lru_cache(maxsize=1)
def check_mongo_connection():
    """Check MongoDB connection once and cache result"""
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        return True
    except ConnectionFailure:
        return False

def clear_session_state_for_chat():
    """Clear chat-specific session state"""
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.chat_history = []

# --- Login / Signup Page ---
def show_login_page():
    # Create centered login container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("Welcome to Legal Case Bot ⚖️")
        
        # Use tabs instead of radio for better UX
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Sign In")
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login", use_container_width=True)
                
                if login_submitted:
                    if login_username and login_password:
                        with st.spinner("Logging in..."):
                            user = login_user(login_username, login_password)
                            if user:
                                st.session_state.logged_in = True
                                st.session_state.user = user
                                st.session_state.current_chat_id = None
                                st.success("Login successful!")
                                time.sleep(0.5)  # Brief pause for UX
                                st.rerun()
                            else:
                                st.error("Invalid username or password.")
                    else:
                        st.error("Please enter both username and password.")

        with tab2:
            st.subheader("Create New Account")
            with st.form("signup_form"):
                signup_username = st.text_input("Username")
                signup_email = st.text_input("Email")
                signup_password = st.text_input("Password", type="password")
                signup_submitted = st.form_submit_button("Sign Up", use_container_width=True)
                
                if signup_submitted:
                    if signup_username and signup_email and signup_password:
                        with st.spinner("Creating account..."):
                            result = signup_user(signup_username, signup_password, signup_email)
                            if "success" in result.lower():
                                st.success(result)
                            else:
                                st.error(result)
                    else:
                        st.error("Please fill in all fields.")

# --- Chat Page ---
def show_chat_page():
    agents_db = get_agents_db()
    
    with st.sidebar:
        st.markdown("<h1 class='sidebar-header'>Legal Case Bot</h1>", unsafe_allow_html=True)

        user = st.session_state.user
        st.markdown(f"**Logged in as:** {user['username']}")

        if st.button("Logout", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("You have been logged out.")
            time.sleep(0.5)
            st.rerun()

        st.markdown("---")

        # --- Case Files Titles ---
        st.subheader("Available Case Files")
        tools = get_tools()  # Use cached tools
        
        # Display case files in an expander to save space
        with st.expander("View Case Files", expanded=False):
            for case in tools.keys():
                st.markdown(f"• {case}")

        st.markdown("---")

        if st.button("➕ New Chat", use_container_width=True):
            clear_session_state_for_chat()
            st.success("New chat started!")
            st.rerun()

        st.markdown("---")
        st.subheader("Chat History")

        # Cache user chats to avoid repeated DB calls
        @st.cache_data(ttl=30)  # Cache for 30 seconds
        def get_cached_user_chats(_user_id):  # Leading underscore tells Streamlit not to hash this param
            return get_user_chats(_user_id, agents_db)

        user_chats = get_cached_user_chats(st.session_state.user["_id"])

        if not user_chats:
            st.info("No chats found.")
        else:
            # Limit displayed chats to prevent overcrowding
            max_chats = 10
            displayed_chats = user_chats[:max_chats]
            
            for chat in displayed_chats:
                chat_id = str(chat["_id"])
                title = chat["title"]
                # Truncate long titles
                display_title = "💬" + title[:30] + "..." if len(title) > 30 else title
                
                if st.button(display_title, key=f"chat_btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = []
                    
                    with st.spinner("Loading chat history..."):
                        st.session_state.chat_history = get_chat_history(chat_id, agents_db)
                        
                        # fetch summary from mongo_convo if exists
                        summary = get_chat_summary(chat_id, agents_db)
                        if summary:
                            st.session_state.chat_history.insert(0, ("system", f"Summary: {summary}"))
                    
                    st.rerun()
            
            if len(user_chats) > max_chats:
                st.info(f"Showing {max_chats} of {len(user_chats)} chats")

    # Chat window
    st.title("Legal Case Assistant")
    st.info("💡 Ask me questions about your legal documents or type 'list cases' to see available cases.")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-message-user">{content}</div>', unsafe_allow_html=True)
            elif role == "assistant":
                st.markdown(f'<div class="chat-message-bot">{content}</div>', unsafe_allow_html=True)
            elif role == "system":
                st.markdown(f'<div class="chat-message-summary">{content}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.current_chat_id:
            with st.spinner("Starting new chat..."):
                chat_id = create_new_chat(st.session_state.user["_id"], agents_db)
                st.session_state.current_chat_id = chat_id

        # Show user message
        st.markdown(f'<div class="chat-message-user">{prompt}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append(("user", prompt))

        # Process via mongo_agents.ask_question (saves to DB too)
        with st.spinner("🤔 Thinking..."):
            try:
                response_text = ask_question(
                    st.session_state.user, 
                    st.session_state.current_chat_id, 
                    prompt, 
                    agents_db
                )

                st.markdown(f'<div class="chat-message-bot">{response_text}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append(("assistant", response_text))
                st.cache_data.clear()  # Refresh sidebar chats
            except Exception as e:
                st.error(f"⚠️ Error processing your request: {e}")
                st.info("Please try again or contact support if the problem persists.")

# --- Main App Logic ---
def main():
    # Initialize session state first
    init_session_state()
    
    # Load CSS
    css_content = load_css()
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    # Check MongoDB connection
    if not check_mongo_connection():
        st.error("❌ Cannot connect to MongoDB. Please check your connection and `MONGO_URI` environment variable.")
        st.info("Make sure MongoDB is running and the connection string is correct.")
        st.stop()
    
    # Load tools once
    if not st.session_state.tools_loaded:
        with st.spinner("Loading case files..."):
            get_tools()  # This will cache the tools
            st.session_state.tools_loaded = True

    # Show appropriate page
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_chat_page()

if __name__ == "__main__":
    main()