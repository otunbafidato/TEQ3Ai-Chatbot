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
import time
import random

# Page configuration
st.set_page_config(
    page_title="TEQ3 AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat container */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: white;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #5568d3;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Input field styling */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 16px;
    }
    
    /* Header styling */
    h1 {
        color: #1a202c;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #2d3748;
    }
    
    /* Info boxes */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    
    /* Success/info messages */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Utility functions
def categorize_query(query):
    """Categorize user queries for smart routing"""
    query_lower = query.lower()
    
    technical_keywords = [
        "payment", "purchase", "buying", "can't buy", "payment failed", 
        "credit card", "billing", "refund", "login", "password", "access",
        "platform", "website", "technical", "error", "bug", "not working",
        "loading", "broken", "system", "account", "sign in", "log in", "can't access"
    ]
    
    complaint_keywords = [
        "complaint", "complain", "issue", "problem", "concern", "worried",
        "dissatisfied", "disappointed", "frustrated", "unhappy", "feedback",
        "poor", "bad", "terrible", "awful", "wrong", "mistake", "unsatisfied"
    ]
    
    career_keywords = [
        "which course", "what should i", "career change", "career transition",
        "job prospects", "salary", "right for me", "help me choose", "confused",
        "not sure", "advice", "recommend", "best option", "career path",
        "switch to tech", "enrollment decision", "course selection", "ai career",
        "data career", "job guarantee", "placement"
    ]
    
    consultant_keywords = [
        "yes", "sure", "interested", "consultant", "career advisor", 
        "speak to someone", "human", "person", "advisor", "guidance",
        "help me decide", "connect me", "arrange", "schedule", "talk to",
        "call me", "consultation"
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

**Our tech support can assist with:**
• Payment and billing problems 💳
• Account access issues 🔐
• Platform technical difficulties 💻
• Purchase and enrollment problems 📝

**Here's how to get help:**
📧 Email: support@teq3.ai
📞 Phone: [Available on our website]
💬 Live Chat: Check teq3.ai for instant support
⏰ Support Hours: We're here when you need us!

For the fastest response, I'd recommend checking our website for live chat options. Is there anything else I can help you with while you're here? 😊"""

def handle_complaint():
    return """I'm really sorry to hear about your concern! 😔 Your feedback is super important to us, and I want to make sure you get the best possible help.

**Let me connect you with the right team:**

🎯 **For course or program concerns:**
📧 careers@teq3.ai | 📞 [Career team number]
They're amazing at addressing program questions and providing personalized guidance!

🔧 **For technical, billing, or platform issues:**
📧 support@teq3.ai | 📞 [Tech support number]
Our tech wizards can sort out any platform problems

📋 **For general feedback or escalated concerns:**
📧 hello@teq3.ai | 📞 [Main contact number]
Direct line to our customer care team

Would you like me to help you get connected with the most appropriate team? I'm here to make sure you get the support you deserve! 💪"""

def handle_career_consultation():
    return """That's fantastic! 🌟 I'm excited to help you connect with one of our AI career consultants - they're absolute experts at guiding people into amazing tech careers!

**Here's how to reach our career consultation team:**
🌐 Visit: teq3.ai/contact
📧 Email: careers@teq3.ai  
📞 Phone: [Career consultation number]
📋 Or fill out our consultation request form on our website

**Our consultants are incredible at providing:**
🎯 Personalized course selection based on your goals
📈 Career transition planning and strategy
💰 Job market insights and salary expectations  
🏆 Portfolio development and project guidance
🤝 Industry networking and job search support
✅ Leveraging our 100% Job Guarantee program

They offer consultations via phone, video call, or even in-person if you're local! 

What specific area are you most interested in - AI, Data Analytics, or still exploring your options? 🤔"""

def suggest_career_consultation():
    suggestions = [
        "💡 Would you like to chat with one of our AI career consultants? They're amazing at helping people find their perfect tech path! 🚀",
        "💡 Our career consultants are like career GPS systems - they can guide you to exactly where you want to go in tech! Want me to connect you? 🗺️",
        "💡 Since this is such an exciting career decision, would you benefit from a personalized chat with our career experts? They love helping people like you! 💡",
        "💡 I can arrange a consultation with our career team - they're fantastic at turning career dreams into reality! Interested? ✨"
    ]
    return random.choice(suggestions)

@st.cache_resource
def initialize_chatbot(api_key):
    """Initialize the chatbot with caching for better performance"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        
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
            except Exception:
                pass
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.8, max_tokens=300, openai_api_key=api_key)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
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
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=custom_prompt
            )}
        )
        
        return chain, True
    except Exception as e:
        return None, str(e)

# Auto-initialize on first load
if not st.session_state.initialized:
    # Try to get API key from secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        with st.spinner("🚀 Initializing your AI assistant..."):
            chain, result = initialize_chatbot(api_key)
            if chain:
                st.session_state.chain = chain
                st.session_state.initialized = True
            else:
                st.error(f"Failed to initialize: {result}")
    except Exception as e:
        st.error("⚠️ Please add your OPENAI_API_KEY to Streamlit secrets")
        st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 TEQ3 AI Assistant")
    st.markdown("---")
    
    # Status indicator
    if st.session_state.initialized:
        st.success("✅ Assistant Ready")
    else:
        st.warning("⏳ Initializing...")
    
    st.markdown("---")
    
    # Quick info section
    st.markdown("### 📚 About TEQ3")
    st.info("""
    **Transform Your Career with AI!**
    
    🤖 AI & Machine Learning  
    📊 Data Analytics  
    🎯 100% Job Guarantee  
    🌟 Industry Experts
    """)
    
    st.markdown("### 📞 Quick Contact")
    st.markdown("""
    📧 hello@teq3.ai  
    🌐 www.teq3.ai  
    💬 careers@teq3.ai  
    🛠️ support@teq3.ai
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Statistics
    st.markdown("### 📊 Chat Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        status = "🟢 Active" if st.session_state.initialized else "🔴 Inactive"
        st.metric("Status", status)

# Main content area
st.markdown("# 🤖 TEQ3 AI Assistant")
st.markdown("### Your Gateway to an AI-Powered Career! 🚀")
st.markdown("---")

# Welcome message for new users
if len(st.session_state.messages) == 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #667eea;'>
            <h3 style='color: #667eea; margin-bottom: 10px;'>🎓 World-Class Training</h3>
            <p style='color: #2d3748; margin: 0;'>Learn AI & Data Analytics from industry experts with hands-on projects</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #667eea;'>
            <h3 style='color: #667eea; margin-bottom: 10px;'>💼 100% Job Guarantee</h3>
            <p style='color: #2d3748; margin: 0;'>We're committed to your success with our job placement guarantee</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #667eea;'>
            <h3 style='color: #667eea; margin-bottom: 10px;'>🌟 Career Support</h3>
            <p style='color: #2d3748; margin: 0;'>Get personalized guidance from our expert career consultants</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here... Ask me anything about TEQ3! 💬"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            query_category = categorize_query(prompt)
            
            # Handle different query types
            if query_category == "consultant_interest":
                response = handle_career_consultation()
            elif query_category == "technical":
                response = handle_technical_support()
            elif query_category == "complaint":
                response = handle_complaint()
            else:
                try:
                    response = st.session_state.chain({"question": prompt})["answer"]
                    
                    if query_category == "career_guidance":
                        response += "\n\n" + suggest_career_consultation()
                except Exception as e:
                    response = f"Oops! 😅 I encountered a hiccup. Let me connect you with support:\n\n{handle_technical_support()}"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 20px;'>
    <p style='margin: 0;'>Made with ❤️ by TEQ3 | Transforming Careers Through AI Education</p>
    <p style='margin: 5px 0 0 0;'>
        <a href='https://www.teq3.ai' style='color: #667eea; text-decoration: none;'>Visit our website</a> | 
        <a href='mailto:hello@teq3.ai' style='color: #667eea; text-decoration: none;'>Contact us</a>
    </p>
</div>
""", unsafe_allow_html=True)
