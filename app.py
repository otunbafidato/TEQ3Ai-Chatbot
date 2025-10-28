import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import random

# Page configuration
st.set_page_config(
    page_title="CareerGPT by TEQ3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black and white theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #ffffff 0%, #e0e0e0 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #000000;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        color: #333333;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Bot message */
    .bot-message {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95);
        color: #000000;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ffffff;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        color: #000000;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #e0e0e0 0%, #ffffff 100%);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #000000 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ffffff;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def categorize_query(query):
    """Categorize user queries for smart routing - only for consultant interest"""
    query_lower = query.lower()
    
    # Only route consultant interest queries - let LLM handle everything else
    consultant_keywords = [
        "speak to someone", "talk to human", "human advisor", "real person",
        "schedule call", "book consultation", "arrange meeting", "connect me with"
    ]
    
    if any(keyword in query_lower for keyword in consultant_keywords):
        return "consultant_interest"
    else:
        return "general"

def handle_technical_support():
    """Provide technical support as LAST RESORT only"""
    return """If none of the solutions above worked, our technical support team is here to help:

📧 Email: support@teq3.ai
💬 Live Chat: Visit teq3.ai for instant support
⏰ Response time: Usually within 2-4 hours

They'll be able to look into your specific account and resolve this quickly! 🛠️"""

def handle_complaint():
    """Handle complaints with empathy and escalation"""
    return """I'm truly sorry you're experiencing this issue. Your satisfaction is important to us.

For urgent concerns that need immediate attention:
📧 careers@teq3.ai (for program/course concerns)
📧 support@teq3.ai (for technical/billing issues)
📧 hello@teq3.ai (for general feedback)

Our team will prioritize your concern and get back to you within 24 hours. 💪"""

def handle_career_consultation():
    """Provide career consultation information"""
    return """That's fantastic! 🌟 I'm excited to help you connect with one of our AI career consultants - they're absolute experts at guiding people into amazing tech careers!

Here's how to reach our career consultation team:
   🌐 Visit: teq3.ai/contact
   📧 Email: careers@teq3.ai  
   📞 Phone: [Career consultation number]
   📋 Or fill out our consultation request form on our website

Our consultants are incredible at providing:
   🎯 Personalized course selection based on your goals
   📈 Career transition planning and strategy
   💰 Job market insights and salary expectations  
   🏆 Portfolio development and project guidance
   🤝 Industry networking and job search support
   ✅ Leveraging our 100% Job Guarantee program

They offer consultations via phone, video call, or even in-person if you're local! 

What specific area are you most interested in - AI, Data Analytics, or still exploring your options? 🤔"""

def suggest_career_consultation():
    """Smart suggestion for career consultation"""
    suggestions = [
        "Would you like to speak with one of our career consultants for personalized guidance? 🚀",
        "Want to connect with our career team for a deeper dive into your options? 🗺️",
        "Interested in a one-on-one consultation to create your personalized roadmap? 💡"
    ]
    return random.choice(suggestions)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi there! 👋 I'm excited to help you with your AI or Data Analytics career journey. What brings you here today? 😊"
    })

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot with caching"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Load documents
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
            except Exception as e:
                st.warning(f"Could not load {url}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=600,
            openai_api_key=api_key
        )
        
       # Create prompt template
custom_prompt = """You are **CareerGPT**, TEQ3's expert AI career advisor specializing in AI and tech industry careers. 
You help users plan, pivot, or level up their careers through personalized, strategic advice. 
Your tone is friendly, knowledgeable, supportive, and concise.

---

## 🎯 Your Core Mission
Help users successfully transition into or advance within AI and tech careers by providing:
- Strategic career planning and path recommendations
- Skill gap analysis and learning roadmaps
- Job role guidance and market insights
- Portfolio, resume, and interview preparation
- Connections to TEQ3's programs and resources

---

## 🧠 When a user reports a problem (e.g., “I can’t buy a course”, “site not loading”, “link not working”):
Follow this **Diagnostic Support Flow** before escalating to human support:

1. **Acknowledge** the problem with empathy and positivity.  
2. **Ask 2–3 friendly diagnostic questions** conversationally to understand what’s happening.  
3. Wait for the user’s reply before suggesting a fix.  
4. Give a **specific, step-by-step solution** for their exact case.  
5. If it doesn’t work, **ask follow-up questions** and try another solution.  
6. After **2–3 attempts**, if it’s still unresolved — **escalate to human support** using this format:

   “It seems this might need our support team’s help. You can reach them at:  
   📧 support@teq3.ai  
   💬 teq3.ai (Live Chat – fastest option)  
   ⏰ They usually respond within 2–4 hours.”

---

## 💬 Example Conversation Flow:

**User:** "I have issue purchasing a course"  
**Bot:** "Oh no, I'm sorry you're having trouble! 😔 Let me help you sort this out.  
Quick question – what exactly happens when you try to buy the course? Do you see an error message, or does your payment get declined?"  

**User:** "Payment declined"  
**Bot:** "Okay, payment declined – that's frustrating but usually fixable! 💳  
Is this happening with a credit card, debit card, or another payment method? And is it a Nigerian card or international?"  

**User:** "Nigerian debit card"  
**Bot:** "Got it! Nigerian debit cards sometimes have restrictions on international online payments.  
Try these two things:  
1️⃣ Call your bank and ask them to enable international online transactions  
2️⃣ Make sure your card is activated for online purchases (some banks require this separately)  
Can you try that and let me know if it works? 🙏"  

**User:** "Still not working"  
**Bot:** "I see, that's definitely frustrating. Let's try one more thing:  
Have you tried using a different card or payment method? Sometimes the issue is specific to one card.  
If you don’t have another option, I can connect you with our payment support team who can help manually. Would that help?"  

**User:** "Yes, connect me"  
**Bot:** "Absolutely! Our support team will get you enrolled right away. 💪  
📧 support@teq3.ai  
💬 teq3.ai (Live Chat – fastest option)  
⏰ Response within 2–4 hours.  
Mention your payment was declined for [course name]. They’ll assist immediately!"

---

Context from TEQ3 website: {context}  
Previous conversation: {chat_history}  
Current question: {question}  

Your response:
"""

        
        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=custom_prompt
        )
        
        # Create chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=False
        )
        
        return chain, True
        
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, False

# Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 CareerGPT by TEQ3</h1>
        <p>Your AI-Powered Career Advisor for Tech Success</p>
    </div>
""", unsafe_allow_html=True)

# Initialize chatbot
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing CareerGPT..."):
        chain, success = initialize_chatbot()
        if success:
            st.session_state.chain = chain
            st.session_state.initialized = True
        else:
            st.error("⚠️ Could not initialize CareerGPT. Please check your API key in Streamlit secrets.")
            st.stop()

# Features section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <h3>💬 Career Planning</h3>
            <p>Get personalized career roadmaps</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <h3>🎓 Course Selection</h3>
            <p>Find your perfect tech path</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <h3>🎯 100% Job Guarantee</h3>
            <p>Land your dream tech job</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="bot-message">
                    <strong>CareerGPT:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)

# Initialize input key counter for clearing
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Input area
st.markdown("<br>", unsafe_allow_html=True)
user_input = st.text_input("💭 Type your message here...", key=f"user_input_{st.session_state.input_key}", placeholder="Ask me about AI careers, courses, or career transitions...")

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_button = st.button("Send 📤", use_container_width=True)

with col2:
    clear_button = st.button("Clear Chat 🗑️", use_container_width=True)

if clear_button:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi there! 👋 I'm excited to help you with your AI or Data Analytics career journey. What brings you here today? 😊"
    })
    st.session_state.input_key += 1
    st.rerun()

if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Categorize query
    query_category = categorize_query(user_input)
    
    # Generate response
    if query_category == "consultant_interest":
        response = handle_career_consultation()
    else:
        # Let the LLM handle everything - it's now trained to troubleshoot
        try:
            response = st.session_state.chain({"question": user_input})["answer"]
        except Exception as e:
            response = f"I encountered an error processing your request. Let me connect you with our support team who can help: {handle_technical_support()}"
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Increment key to clear input field
    st.session_state.input_key += 1
    st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: rgba(255, 255, 255, 0.5); padding: 2rem;">
        <p>🌟 Powered by TEQ3 AI | Visit <a href="https://www.teq3.ai" style="color: #ffffff;">teq3.ai</a> to start your journey</p>
    </div>
""", unsafe_allow_html=True)
