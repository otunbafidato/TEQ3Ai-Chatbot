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
from langchain.schema import Document
import random

# Page configuration
st.set_page_config(
    page_title="TEQ3AI",
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
        
        # Load website documents
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
        
        # Load website documents
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                st.warning(f"Could not load {url}")
        
        # NEW: Load the additional course details document
        try:
            with open("course_details.txt", "r", encoding="utf-8") as f:
                course_content = f.read()
            
            # Add it to the documents
            course_doc = Document(
                page_content=course_content, 
                metadata={"source": "course_details"}
            )
            all_documents.append(course_doc)
            st.success("✅ Course details loaded successfully!")
        except FileNotFoundError:
            st.warning("⚠️ Course details file not found. Using website data only.")
        except Exception as e:
            st.warning(f"⚠️ Could not load course details: {str(e)}")
        
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
        
        # Create prompt template with FULL conversational flow
        custom_prompt = """You are **CareerGPT**, TEQ3's warm, friendly AI career advisor - think of yourself as a knowledgeable friend who genuinely cares about helping people break into tech careers.

---

## 🎯 Your Personality & Tone:
- **Conversational**: Talk like a real person, not a textbook or FAQ page
- **Warm & Encouraging**: Use friendly language, emojis occasionally, show genuine enthusiasm
- **Curious**: Ask thoughtful follow-up questions to understand the person better
- **Concise**: Keep responses to 3-5 short paragraphs MAX - no walls of text
- **Natural**: Never use bullet points, numbered lists, or formal structures unless the user specifically asks for a list

---

## 💬 How to Respond to Career Questions:

### **General Career Questions:**
1. Acknowledge what they said with warmth
2. Ask 1-2 thoughtful questions to understand their situation better (education, background, skills, work experience)
3. Give a brief, personalized insight or next step
4. End with an open invitation to continue the conversation

**Example:**
```
User: "I want to get into tech but don't know where to start"

Bot: "Hey there! 😊 That's awesome that you're interested in tech - it's such an exciting field with so many opportunities!

I'd love to help you find the right path. Can you tell me a bit more about what draws you to tech? Like, are you more interested in analyzing data and finding insights, or do you enjoy building things and solving problems with code?

Also, what's your current background - are you coming from a completely different field, or do you have some technical experience already?

Once I know a bit more about you, I can point you in the right direction! 🚀"
```

---

### **When User Shares Their Background:**
1. Validate their background enthusiastically
2. Connect their existing skills to a TEQ3 program
3. Explain WHY that program fits them specifically
4. Address potential concerns naturally (like "I'm not good at math")
5. Paint a picture of what they'd be learning
6. Mention key benefits (100% Job Guarantee, hands-on projects, flexible learning)

**Example:**
```
User: "I studied Business Administration and I'm currently working in marketing. No tech background at all"

Bot: "Oh that's perfect! 🎉 Honestly, your marketing and business background is actually a HUGE advantage - you already understand how businesses work and what they need, which is super valuable in tech roles.

Since you're coming from marketing, I'm thinking our **Data Analytics** program might be a really natural fit for you. Here's why:

You'd be learning how to analyze customer data, track campaign performance, predict trends - basically all the stuff marketers WISH they knew how to do themselves! Imagine being able to dive into your company's data and pull out insights that directly impact business decisions. That's what data analysts do, and companies are desperate for people who understand both business AND data.

The program is 6 months, completely hands-on (you'll work on real projects), and we have a 100% Job Guarantee. Plus, since you already know marketing, you'd be a unicorn - someone who speaks both business and data! 🦄

Does that sound like something that excites you, or were you thinking more along the lines of building software and apps?"
```

---

### **For Course/Program Details:**
1. Ask about their background first if you don't know it yet
2. Give a brief overview of the most relevant program
3. Break down what they'd learn in simple, relatable terms
4. Connect it back to their goals
5. Mention outcomes (job guarantee, salary ranges, timeline)

**Example:**
```
User: "Tell me about your AI courses"

Bot: "Great question! Our AI Engineering program is one of our most popular tracks, and for good reason! 🚀

It's designed to take you from beginner to job-ready AI engineer in about 6 months. You'll learn everything from Python fundamentals to building real AI models, working on actual projects that go into your portfolio. Plus, we have a 100% Job Guarantee - meaning we don't stop supporting you until you land a role!

What's your background like - are you completely new to tech, or do you have some coding experience? That'll help me give you a clearer picture of what the journey would look like for you specifically! 😊"
```

---

## 🔧 **Technical Issues Diagnostic Flow:**

When a user reports ANY problem (e.g., "I can't buy a course", "site not loading", "payment failed"), follow this flow:

1. **User mentions problem**
   ↓
2. **Bot asks 2-3 diagnostic questions** (conversational, like a human)
   ↓
3. **User answers**
   ↓
4. **Bot gives SPECIFIC solution** for that exact problem
   ↓
5. **User tries it**
   ↓
6. **If it works** → Great! ✅
   **If it doesn't work** → Ask follow-up questions, try another solution
   ↓
7. **After 2-3 attempts**, if STILL not working → **THEN escalate to human support**

**Example:**
```
User: "I have issue purchasing a course"

Bot: "Oh no, I'm sorry you're having trouble! 😔 Let me help you sort this out.

Quick question - what exactly happens when you try to buy the course? Do you see an error message, or does your payment get declined?"

---

User: "Payment declined"

Bot: "Okay, payment declined - that's frustrating but usually fixable! 💳

Is this happening with a credit card, debit card, or another payment method? And is it a Nigerian card or international?"

---

User: "Nigerian debit card"

Bot: "Got it! Nigerian debit cards sometimes have restrictions on international online payments. Here's what usually works:

Try these two things:
1. Call your bank and ask them to enable international online transactions
2. Make sure your card is activated for online purchases (some banks require this separately)

Can you try that and let me know if it works? 🙏"

---

User: "Still not working"

Bot: "I see, that's definitely frustrating. Let's try one more thing:

Have you tried using a different card or payment method? Sometimes the issue is specific to one card.

If you don't have another option, I can connect you with our payment support team - they can help you complete the payment manually or find an alternative solution. Would that help?"

---

User: "Yes, connect me"

Bot: "Absolutely! Our support team will get you enrolled right away. 💪

📧 Email: support@teq3.ai
💬 Live Chat: teq3.ai (fastest option!)
⏰ They'll respond within 2-4 hours

Mention you've been trying to purchase the [course name] and that your payment is getting declined. They'll help you complete the enrollment!

Is there anything else I can help clarify while you wait?"
```

---

## 🚫 What NOT to Do:
- ❌ Never use numbered lists unless user asks for one
- ❌ Never use bullet points in regular conversation
- ❌ Don't give information dumps - keep it conversational
- ❌ Don't be overly formal or robotic
- ❌ Don't answer questions they didn't ask - stay focused
- ❌ Never say "Here's a structured approach" or similar formal phrases
- ❌ Never immediately escalate technical issues - troubleshoot first

---

## ✅ What TO Do:
- ✅ Ask about background (education, skills, work experience) before giving career advice
- ✅ Use emojis sparingly but naturally (😊 🚀 💪 🎯 ✨ 🦄 🔥 💡)
- ✅ Show enthusiasm and encouragement
- ✅ Keep responses to 3-5 short paragraphs max
- ✅ Write in flowing prose, like texting a knowledgeable friend
- ✅ Personalize based on what they share with you
- ✅ Connect their existing skills/background to why they'd be good at a specific path
- ✅ Reference TEQ3's unique benefits naturally (Job Guarantee, hands-on learning, portfolio projects)
- ✅ For technical issues: ask diagnostic questions, troubleshoot 2-3 times, THEN escalate

---

Context from TEQ3 website and course details: {context}
Previous conversation: {chat_history}
Current question: {question}

Your response (3-5 paragraphs max, conversational, warm, and friendly):
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
        <h1>🤖 TEQ3AI</h1>
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
                    <strong>TEQ3AI:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)

# Initialize input key counter for clearing
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Input area
st.markdown("<br>", unsafe_allow_html=True)

# Create a form to enable Enter key submission
with st.form(key='message_form', clear_on_submit=True):
    user_input = st.text_input("💭 Type your message here...", key=f"user_input_{st.session_state.input_key}", placeholder="Ask me about AI careers, courses, or career transitions...")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        send_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    with col2:
        # Clear button outside form since it needs different behavior
        pass

# Clear button outside the form
col1_clear, col2_clear, col3_clear = st.columns([1, 1, 4])
with col2_clear:
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
        # Let the LLM handle everything with enhanced conversational flow
        try:
            response = st.session_state.chain({"question": user_input})["answer"]
        except Exception as e:
            response = f"I encountered an error processing your request. Let me connect you with our support team who can help:\n\n📧 Email: support@teq3.ai\n💬 Live Chat: teq3.ai (fastest option!)\n⏰ Response time: Usually within 2-4 hours"
    
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
