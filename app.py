import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import openai
import random
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="TEQ3 AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black theme and beautiful UI
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Messages styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: rgba(40, 40, 40, 0.95);
        color: #ffffff;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            transform: translateX(20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInLeft {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(30, 30, 30, 0.95);
        color: white;
        border: 1px solid rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 20, 20, 0.95);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(74, 222, 128, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(74, 222, 128, 0);
        }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 15px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        margin: 0 3px;
        animation: typing 1.5s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.5;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.memory = None
    st.session_state.chain = None
    st.session_state.vectorstore = None
    st.session_state.api_key_valid = False

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🤖 TEQ3 AI Assistant</div>
    <div class="header-subtitle">Your Gateway to AI & Data Analytics Careers | 100% Job Guarantee 🚀</div>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to start chatting"
    )
    
    if api_key:
        # Initialize button
        if st.button("🚀 Initialize Chatbot", use_container_width=True):
            with st.spinner("🔄 Initializing TEQ3 AI Assistant..."):
                try:
                    # Set API key
                    openai.api_key = api_key
                    os.environ["OPENAI_API_KEY"] = api_key
                    
                    # Initialize the chatbot
                    @st.cache_resource
                    def initialize_chatbot(api_key):
                        # Load web content
                        urls = [
                            "https://www.teq3.ai/",
                            "https://www.teq3.ai/about-us",
                            "https://www.teq3.ai/courses",
                            "https://www.teq3.ai/programs",
                            "https://www.teq3.ai/data-analytics",
                            "https://www.teq3.ai/artificial-intelligence",
                            "https://www.teq3.ai/contact",
                            "https://www.teq3.ai/services"
                        ]
                        
                        all_documents = []
                        for url in urls:
                            try:
                                loader = WebBaseLoader(url)
                                documents = loader.load()
                                all_documents.extend(documents)
                            except:
                                pass
                        
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        texts = text_splitter.split_documents(all_documents)
                        
                        # Create embeddings and vector store
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        vectorstore = FAISS.from_documents(texts, embeddings)
                        
                        # Initialize LLM
                        llm = OpenAI(
                            model_name="gpt-3.5-turbo-instruct",
                            temperature=0.8,
                            max_tokens=300,
                            openai_api_key=api_key
                        )
                        
                        # Set up memory
                        memory = ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True
                        )
                        
                        # Create prompt template
                        custom_prompt = """
You are TEQ3's AI Assistant - a friendly, helpful, and intuitive chatbot that acts like ChatGPT but specializes in TEQ3's AI and Data Analytics training programs.

PERSONALITY TRAITS:
- Be conversational, warm, and genuinely helpful
- Show enthusiasm about AI and data careers
- Use emojis occasionally to be friendly 😊
- Ask follow-up questions to better understand user needs
- Be encouraging and supportive about career transitions
- Provide detailed, thoughtful responses like ChatGPT would

CAPABILITIES:
- Answer questions about TEQ3's courses, programs, and services
- Provide career guidance and advice
- Help with general AI and data science questions
- Assist with enrollment and program selection
- Handle technical support routing
- Engage in friendly conversation while staying helpful

RESPONSE STYLE:
- Start with a warm greeting or acknowledgment
- Provide comprehensive, helpful answers
- End with a relevant follow-up question or offer of assistance
- Be conversational and natural, not robotic
- Show personality while remaining professional

If you don't know something specific about TEQ3, be honest and offer to connect them with the right person.

Context: {context}
Chat History: {chat_history}
User: {question}

TEQ3 AI Assistant:
"""
                        
                        # Create chain
                        chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                            memory=memory,
                            combine_docs_chain_kwargs={
                                "prompt": PromptTemplate(
                                    input_variables=["context", "chat_history", "question"],
                                    template=custom_prompt
                                )
                            }
                        )
                        
                        return chain, memory, vectorstore
                    
                    chain, memory, vectorstore = initialize_chatbot(api_key)
                    st.session_state.chain = chain
                    st.session_state.memory = memory
                    st.session_state.vectorstore = vectorstore
                    st.session_state.initialized = True
                    st.session_state.api_key_valid = True
                    
                    # Add welcome message
                    if not st.session_state.messages:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Hi there! 👋 Welcome to TEQ3! I'm your AI Assistant, and I'm super excited to help you explore our amazing AI & Data Analytics programs! 🚀\n\nWe offer:\n• 🤖 Artificial Intelligence courses\n• 📊 Data Analytics programs\n• 🎯 100% Job Guarantee\n• 💼 Career transition support\n\nWhat brings you here today? Are you looking to start a career in AI or Data Science? I'm here to help! 😊"
                        })
                    
                    st.success("✅ Chatbot initialized successfully!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error initializing chatbot: {str(e)}")
                    st.session_state.api_key_valid = False
    
    # Display status
    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.initialized:
        st.markdown('<span class="status-online"></span> **Online**', unsafe_allow_html=True)
        st.success("✅ Ready to chat!")
    else:
        st.warning("⚠️ Please enter API key and initialize")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Courses", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.messages.append({"role": "user", "content": "Tell me about your courses"})
                st.rerun()
    
    with col2:
        if st.button("💼 Careers", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.messages.append({"role": "user", "content": "I want career guidance"})
                st.rerun()
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🎯 Job Guarantee", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.messages.append({"role": "user", "content": "Tell me about the job guarantee"})
                st.rerun()
    
    with col4:
        if st.button("📞 Contact", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.messages.append({"role": "user", "content": "How can I contact TEQ3?"})
                st.rerun()
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ Information")
    st.markdown("""
    <div class="info-box">
        <strong>🌐 Website:</strong> teq3.ai<br>
        <strong>📧 Support:</strong> support@teq3.ai<br>
        <strong>📧 Careers:</strong> careers@teq3.ai<br>
        <strong>⏰ Available:</strong> 24/7 AI Support
    </div>
    """, unsafe_allow_html=True)
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.memory:
            st.session_state.memory.clear()
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat cleared! 🔄 How can I help you today? 😊"
        })
        st.rerun()

# Query analysis functions
def categorize_query(query):
    """Categorize user queries for smart routing"""
    query_lower = query.lower()
    
    technical_keywords = [
        "payment", "purchase", "buying", "can't buy", "payment failed",
        "credit card", "billing", "refund", "login", "password", "access",
        "platform", "website", "technical", "error", "bug", "not working"
    ]
    
    complaint_keywords = [
        "complaint", "complain", "issue", "problem", "concern", "worried",
        "dissatisfied", "disappointed", "frustrated", "unhappy"
    ]
    
    career_keywords = [
        "which course", "what should i", "career change", "career transition",
        "job prospects", "salary", "right for me", "help me choose"
    ]
    
    consultant_keywords = [
        "yes", "sure", "interested", "consultant", "career advisor",
        "speak to someone", "human", "person", "advisor"
    ]
    
    if any(keyword in query_lower for keyword in technical_keywords):
        return "technical"
    elif any(keyword in query_lower for keyword in complaint_keywords):
        return "complaint"
    elif any(keyword in query_lower for keyword in career_keywords):
        return "career_guidance"
    elif any(keyword in query_lower for keyword in consultant_keywords):
        return "consultant_interest"
    else:
        return "general"

def handle_technical_support():
    return """I understand you're experiencing a technical issue! 🛠️ Don't worry, our technical support team is here to help.

Our tech support can assist with:
• Payment and billing problems 💳
• Account access issues 🔐
• Platform technical difficulties 💻
• Purchase and enrollment problems 📝

Here's how to get help:
📧 Email: support@teq3.ai
📞 Phone: Check our website for the latest number
💬 Live Chat: Visit teq3.ai for instant support
⏰ Support Hours: We're here when you need us!

Is there anything else I can help you with while you're here? 😊"""

def handle_complaint():
    return """I'm really sorry to hear about your concern! 😔 Your feedback is super important to us, and I want to make sure you get the best possible help.

Let me connect you with the right team:

🎯 **For course or program concerns:**
📧 careers@teq3.ai
They're amazing at addressing program questions!

🔧 **For technical, billing, or platform issues:**
📧 support@teq3.ai
Our tech wizards can sort out any problems

📋 **For general feedback:**
📧 hello@teq3.ai
Direct line to our customer care team

Would you like me to help you get connected with the most appropriate team? 💪"""

def handle_career_consultation():
    return """That's fantastic! 🌟 I'm excited to help you connect with one of our AI career consultants!

Here's how to reach our career consultation team:
🌐 Visit: teq3.ai/contact
📧 Email: careers@teq3.ai
📞 Phone: Check our website for the number
📋 Or fill out our consultation request form online

Our consultants provide:
🎯 Personalized course selection
📈 Career transition planning
💰 Job market insights
🏆 Portfolio development
🤝 Industry networking support
✅ 100% Job Guarantee program details

They offer consultations via phone, video call, or in-person!

What area interests you most - AI, Data Analytics, or still exploring? 🤔"""

# Main chat interface
if st.session_state.initialized:
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">TEQ3 AI: {message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Type your message...",
                key="user_input",
                placeholder="Ask me anything about TEQ3, AI careers, or our programs! 🚀",
                label_visibility="hidden"
            )
        with col2:
            send_button = st.button("Send 📤", use_container_width=True)
    
    # Process user input
    if (user_input and send_button) or (user_input and st.session_state.get('enter_pressed')):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator
        with st.spinner("TEQ3 AI is typing..."):
            # Categorize query
            query_category = categorize_query(user_input)
            
            # Handle different query types
            if query_category == "consultant_interest":
                response = handle_career_consultation()
            elif query_category == "technical":
                response = handle_technical_support()
            elif query_category == "complaint":
                response = handle_complaint()
            else:
                # Use the LLM chain for general queries
                try:
                    result = st.session_state.chain({"question": user_input})
                    response = result["answer"]
                    
                    # Add smart follow-ups for career guidance
                    if query_category == "career_guidance":
                        follow_ups = [
                            "\n\n💡 Would you like to chat with one of our AI career consultants? They're amazing at helping people find their perfect tech path! 🚀",
                            "\n\n💡 Our career consultants can guide you to exactly where you want to go in tech! Want me to connect you? 🗺️",
                            "\n\n💡 Would you benefit from a personalized chat with our career experts? They love helping people like you! ✨"
                        ]
                        response += random.choice(follow_ups)
                        
                except Exception as e:
                    response = """Oops! 😅 I encountered a little hiccup processing that. No worries though! 
                    
Let me connect you with our amazing support team:
📧 Email: support@teq3.ai
💬 Live Chat: Visit teq3.ai
                    
Is there anything else I can help you with? 😊"""
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear input and refresh
            time.sleep(0.5)
            st.rerun()

else:
    # Show initialization prompt
    st.markdown("""
    <div class="chat-container" style="text-align: center; padding: 50px;">
        <h2>🔐 Please Initialize the Chatbot</h2>
        <p>Enter your OpenAI API key in the sidebar and click "Initialize Chatbot" to start chatting!</p>
        <p>Don't have an API key? Get one from <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI</a></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 50px; color: #888;">
    <p>Powered by TEQ3.AI | Transform Your Career with AI & Data Analytics | 100% Job Guarantee 🚀</p>
</div>
""", unsafe_allow_html=True)