# StudyMate Pro - AI-Powered Study Companion
# File: app.py
# Run with: streamlit run app.py

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer  # Fixed import
import requests
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class StudyMateProBackend:
    def __init__(self):  # Fixed constructor name
        # Initialize all attributes in constructor
        self.embedder = None
        self.vector_store = None
        self.chunks = []
        self.metadata = []
        self.user_data = {}
        self.study_sessions = []
        self.flashcards = []
        self.achievements = []
        
    def initialize_embedder(self):
        """Initialize the sentence transformer model"""
        if self.embedder is None:
            with st.spinner("🧠 Initializing AI Brain..."):
                try:
                    self.embedder = SentenceTransformer(EMBEDDING_MODEL)  # Fixed class name
                except Exception as e:
                    st.error(f"Error initializing embedder: {str(e)}")
                    # Fallback - you can add alternative embedding method here
                    return False
        return True
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, List[Tuple[int, str]]]:
        """Extract text from uploaded PDF"""
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                page_texts.append((page_num + 1, text))
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            doc.close()
            return full_text, page_texts
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return "", []
    
    def create_chunks(self, text: str, page_texts: List[Tuple[int, str]]) -> List[Dict]:
        """Create overlapping chunks with metadata"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            page_num = self._find_page_for_chunk(chunk_text, page_texts)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'page_num': page_num,
                'word_start': i,
                'word_end': min(i + CHUNK_SIZE, len(words))
            })
        
        return chunks
    
    def _find_page_for_chunk(self, chunk_text: str, page_texts: List[Tuple[int, str]]) -> int:
        """Find the page number that best matches the chunk"""
        best_page = 1
        max_overlap = 0
        chunk_words = set(chunk_text.lower().split())
        
        for page_num, page_text in page_texts:
            page_words = set(page_text.lower().split())
            overlap = len(chunk_words.intersection(page_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_page = page_num
        
        return best_page
    
    def create_vector_store(self, chunks: List[Dict]):
        """Create FAISS vector store from chunks"""
        # Initialize embedder first
        if not self.initialize_embedder():
            st.error("Failed to initialize embedder. Cannot create vector store.")
            return
            
        texts = [chunk['text'] for chunk in chunks]
        
        with st.spinner("🔍 Creating Knowledge Base..."):
            try:
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
                dimension = embeddings.shape[1]
                self.vector_store = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(embeddings)
                self.vector_store.add(embeddings.astype('float32'))
                self.chunks = chunks
                self.metadata = chunks
                st.success("✅ Vector store created successfully!")
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar chunks using semantic similarity"""
        if self.vector_store is None or self.embedder is None:
            return []
        
        try:
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
        except Exception as e:
            st.error(f"Error searching chunks: {str(e)}")
            return []
    
    def generate_flashcards(self, text: str, num_cards: int = 5) -> List[Dict]:
        """Generate flashcards from text content"""
        sentences = text.split('.')
        flashcards = []
        
        # Simple flashcard generation logic
        for i, sentence in enumerate(sentences[:num_cards]):
            if len(sentence.strip()) > 20:
                words = sentence.strip().split()
                if len(words) > 5:
                    # Create a fill-in-the-blank style flashcard
                    key_word_idx = len(words) // 2
                    key_word = words[key_word_idx]
                    
                    question = sentence.replace(key_word, "_")
                    answer = key_word
                    
                    flashcards.append({
                        'id': f"fc_{i}",
                        'question': f"Fill in the blank: {question}",
                        'answer': answer,
                        'difficulty': 'medium',
                        'created_at': datetime.now().isoformat()
                    })
        
        return flashcards
    
    def generate_quiz(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate quiz questions from text content"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        quiz_questions = []
        
        for i, sentence in enumerate(sentences[:num_questions]):
            words = sentence.split()
            if len(words) > 8:
                # Create multiple choice questions
                key_concept = words[len(words)//3:len(words)//2]
                key_phrase = " ".join(key_concept)
                
                quiz_questions.append({
                    'id': f"q_{i}",
                    'question': f"What concept is described by: '{sentence[:50]}...'?",
                    'options': [key_phrase, "Alternative A", "Alternative B", "Alternative C"],
                    'correct_answer': 0,
                    'explanation': sentence,
                    'difficulty': 'medium'
                })
        
        return quiz_questions
    
    def track_study_session(self, duration: int, topic: str, performance: float):
        """Track user study session"""
        session = {
            'date': datetime.now().isoformat(),
            'duration': duration,
            'topic': topic,
            'performance': performance,
            'id': len(self.study_sessions)
        }
        self.study_sessions.append(session)
    
    def get_study_analytics(self) -> Dict:
        """Generate study analytics"""
        if not self.study_sessions:
            return {}
        
        df = pd.DataFrame(self.study_sessions)
        analytics = {
            'total_sessions': len(self.study_sessions),
            'total_study_time': sum(session['duration'] for session in self.study_sessions),
            'avg_performance': np.mean([session['performance'] for session in self.study_sessions]),
            'sessions_this_week': len([s for s in self.study_sessions 
                                    if datetime.fromisoformat(s['date']) > datetime.now() - timedelta(days=7)])
        }
        
        return analytics

def create_custom_css():
    """Create custom CSS for ocean blue theme"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --ocean-blue: #006994;
        --light-ocean-blue: #B0D6F5;
        --deep-ocean: #004d6b;
        --wave-white: #ffffff;
        --pearl-gray: #f8f9fc;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #B0D6F5 0%, #87CEEB 50%, #4682B4 100%);
    }
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 105, 148, 0.2);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #006994 0%, #4682B4 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 105, 148, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #B0D6F5 0%, #87CEEB 100%);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        color: #004d6b;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0, 105, 148, 0.2);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        border-left: 4px solid #006994;
        box-shadow: 0 2px 10px rgba(0, 105, 148, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #006994 0%, #4682B4 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 105, 148, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 105, 148, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #B0D6F5 0%, #87CEEB 100%);
    }
    
    /* Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 105, 148, 0.15);
        border: 1px solid rgba(0, 105, 148, 0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 105, 148, 0.25);
    }
    
    /* Flashcard */
    .flashcard {
        background: linear-gradient(135deg, #ffffff 0%, #B0D6F5 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 25px rgba(0, 105, 148, 0.2);
        border: 2px solid transparent;
    }
    
    .flashcard:hover {
        border-color: #006994;
        transform: rotateY(5deg);
    }
    
    /* Progress bar */
    .progress-bar {
        background: linear-gradient(90deg, #006994 0%, #4682B4 100%);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 15px rgba(0, 105, 148, 0.1);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #B0D6F5;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #006994;
        box-shadow: 0 0 10px rgba(0, 105, 148, 0.3);
    }
    
    /* Animations */
    @keyframes wave {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .wave-emoji {
        animation: wave 2s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Motivational quotes */
    .motivation-box {
        background: linear-gradient(135deg, #006994 0%, #4682b4 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-style: italic;
        box-shadow: 0 4px 20px rgba(0, 105, 148, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="StudyMate Pro 🌊",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    create_custom_css()
    
    # Initialize backend
    if 'studymate' not in st.session_state:
        st.session_state.studymate = StudyMateProBackend()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'chat'
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': 'Student',
            'study_streak': 0,
            'total_study_time': 0,
            'achievements': []
        }
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1><span class="wave-emoji">🌊</span> StudyMate Pro <span class="wave-emoji">🧠</span></h1>
        <p style="font-size: 1.2rem; margin: 0;"><em>"Your AI-Powered Study Companion"</em></p>
        <p style="font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0 0;">Upload • Learn • Excel • Achieve</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        mode = st.selectbox(
            "Choose Mode:",
            ["💬 Chat & Learn", "📚 Flashcards", "🎯 Quiz Mode", "📊 Analytics", "⚙ Settings"],
            key="mode_selector"
        )
        
        # Update current mode
        if mode == "💬 Chat & Learn":
            st.session_state.current_mode = 'chat'
        elif mode == "📚 Flashcards":
            st.session_state.current_mode = 'flashcards'
        elif mode == "🎯 Quiz Mode":
            st.session_state.current_mode = 'quiz'
        elif mode == "📊 Analytics":
            st.session_state.current_mode = 'analytics'
        else:
            st.session_state.current_mode = 'settings'
        
        st.markdown("---")
        
        # File Upload Section
        st.markdown("### 📁 Upload Study Materials")
        uploaded_file = st.file_uploader(
            "Choose your file:",
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, text, or Word documents"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file)
        
        st.markdown("---")
        
        # User Profile
        st.markdown("### 👤 Your Profile")
        profile = st.session_state.user_profile
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Welcome, {profile['name']}! 👋</h4>
            <p>🔥 Study Streak: {profile['study_streak']} days</p>
            <p>⏰ Total Time: {profile['total_study_time']} mins</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Motivational Quote
        quotes = [
            "Every expert was once a beginner! 🌟",
            "Success is the sum of small efforts! 💪",
            "Knowledge is power! 🧠⚡",
            "You're making great progress! 🚀",
            "Learning never exhausts the mind! 📖"
        ]
        
        st.markdown(f"""
        <div class="motivation-box">
            {np.random.choice(quotes)}
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    if st.session_state.current_mode == 'chat':
        render_chat_mode()
    elif st.session_state.current_mode == 'flashcards':
        render_flashcard_mode()
    elif st.session_state.current_mode == 'quiz':
        render_quiz_mode()
    elif st.session_state.current_mode == 'analytics':
        render_analytics_mode()
    else:
        render_settings_mode()

def process_uploaded_file(uploaded_file):
    """Process uploaded file and update the knowledge base"""
    try:
        # Create a hash of the file content
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)  # Reset file pointer
        
        if 'current_file_hash' not in st.session_state or st.session_state.current_file_hash != file_hash:
            st.session_state.current_file_hash = file_hash
            
            with st.spinner("📄 Processing your document..."):
                if uploaded_file.type == "application/pdf":
                    full_text, page_texts = st.session_state.studymate.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    # Handle text files
                    full_text = str(file_content, "utf-8")
                    page_texts = [(1, full_text)]
                else:
                    st.error("Unsupported file type. Please upload PDF or TXT files.")
                    return
                
                if full_text and full_text.strip():
                    chunks = st.session_state.studymate.create_chunks(full_text, page_texts)
                    if chunks:
                        st.session_state.studymate.create_vector_store(chunks)
                        st.success(f"✅ Processed {len(chunks)} chunks successfully!")
                        
                        # Store document for future reference
                        if 'uploaded_docs' not in st.session_state:
                            st.session_state.uploaded_docs = []
                        
                        st.session_state.uploaded_docs.append({
                            'name': uploaded_file.name,
                            'upload_time': datetime.now().isoformat(),
                            'chunks': len(chunks),
                            'content': full_text[:500] + "..." if len(full_text) > 500 else full_text
                        })
                    else:
                        st.error("No content could be extracted from the file.")
                else:
                    st.error("The uploaded file appears to be empty or unreadable.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

def render_chat_mode():
    """Render the main chat interface"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat with StudyMate")
        
        # Chat input
        if not hasattr(st.session_state.studymate, 'chunks') or not st.session_state.studymate.chunks:
            st.markdown("""
            <div class="feature-card" style="text-align: center;">
                <h3>🌊 Ready to Dive In? 🌊</h3>
                <p>Upload a document to start your learning journey!</p>
                <p>I can help you with:</p>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>📖 Understanding complex concepts</li>
                    <li>📝 Creating study notes</li>
                    <li>🎯 Generating practice questions</li>
                    <li>💡 Explaining difficult topics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Question input
            question = st.text_input(
                "Ask me anything about your document:",
                placeholder="e.g., Explain the main concepts in simple terms...",
                key="chat_input"
            )
            
            col_ask, col_example = st.columns([1, 2])
            
            with col_ask:
                if st.button("🚀 Ask StudyMate", type="primary"):
                    if question:
                        handle_chat_question(question)
            
            with col_example:
                example_prompts = [
                    "Summarize the key points",
                    "Create flashcards from this content",
                    "Explain this in simple terms",
                    "Generate practice questions"
                ]
                
                selected_example = st.selectbox("Or try an example:", [""] + example_prompts)
                if selected_example and st.button("Use Example"):
                    handle_chat_question(selected_example)
        
        # Display chat history
        render_chat_history()
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if hasattr(st.session_state.studymate, 'chunks') and st.session_state.studymate.chunks:
            if st.button("📚 Generate Flashcards", key="quick_flashcards"):
                generate_flashcards_from_content()
            
            if st.button("🎯 Create Quiz", key="quick_quiz"):
                generate_quiz_from_content()
            
            if st.button("📊 Study Summary", key="quick_summary"):
                generate_study_summary()
        
        # Recent activity
        st.markdown("### 📈 Recent Activity")
        if st.session_state.chat_history:
            recent_topics = list(set([chat['question'][:30] + "..." for chat in st.session_state.chat_history[-3:]]))
            for topic in recent_topics:
                st.markdown(f"• {topic}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_chat_question(question: str):
    """Handle user question and generate response"""
    with st.spinner("🤔 Thinking..."):
        relevant_chunks = st.session_state.studymate.search_similar_chunks(question, k=3)
        
        # Generate contextual answer
        if relevant_chunks:
            context = " ".join([chunk['text'] for chunk in relevant_chunks[:2]])
            answer = generate_smart_answer(question, context)
        else:
            answer = "I couldn't find specific information about that in your document. Could you try rephrasing your question?"
        
        # Add to chat history
        st.session_state.chat_history.append({
            'question': question,
            'answer': answer,
            'chunks': relevant_chunks,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'type': 'qa'
        })
        
        # Track study session
        st.session_state.studymate.track_study_session(
            duration=2,  # Assume 2 minutes per question
            topic=question[:50],
            performance=0.8  # Default performance
        )
    
    st.rerun()

def generate_smart_answer(question: str, context: str) -> str:
    """Generate intelligent answer based on question and context"""
    # Simple rule-based answer generation
    question_lower = question.lower()
    
    if "summary" in question_lower or "summarize" in question_lower:
        sentences = context.split('.')[:3]
        return f"Here's a summary: {'. '.join(sentences)}."
    
    elif "flashcard" in question_lower:
        return "I can create flashcards for you! Switch to Flashcard mode to see them, or I can generate some key points here."
    
    elif "explain" in question_lower or "what is" in question_lower:
        return f"Based on your document: {context[:300]}..."
    
    else:
        return f"Based on the relevant content I found: {context[:400]}..."

def render_chat_history():
    """Render chat conversation history"""
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation")
        
        for chat in reversed(st.session_state.chat_history[-5:]):  # Show last 5 chats
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>🙋 You ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="bot-message">
                <strong>🌊 StudyMate:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Source info
            if chat.get('chunks'):
                best_chunk = chat['chunks'][0]
                st.markdown(f"""
                <div style="font-size: 0.8em; color: #666; margin-bottom: 1rem;">
                    📍 Source: Page {best_chunk['page_num']} (Confidence: {best_chunk['similarity_score']:.2f})
                </div>
                """, unsafe_allow_html=True)

def render_flashcard_mode():
    """Render flashcard creation and study mode"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("### 📚 Flashcard Studio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not hasattr(st.session_state, 'flashcards'):
            st.session_state.flashcards = []
        
        if st.button("🎴 Generate New Flashcards"):
            if hasattr(st.session_state.studymate, 'chunks') and st.session_state.studymate.chunks:
                generate_flashcards_from_content()
        
        # Display flashcards
        if st.session_state.flashcards:
            st.markdown(f"### 🎯 Your Flashcards ({len(st.session_state.flashcards)} cards)")
            
            for i, card in enumerate(st.session_state.flashcards):
                with st.expander(f"Card {i+1}: {card['question'][:50]}..."):
                    col_q, col_a = st.columns(2)
                    
                    with col_q:
                        st.markdown(f"""
                        <div class="flashcard">
                            <h4>Question:</h4>
                            <p>{card['question']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_a:
                        if st.button(f"Show Answer", key=f"show_ans_{i}"):
                            st.markdown(f"""
                            <div class="flashcard">
                                <h4>Answer:</h4>
                                <p style="color: #006994; font-weight: bold;">{card['answer']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("📝 No flashcards yet. Upload a document and generate some!")
    
    with col2:
        st.markdown("### 🎯 Study Controls")
        
        if st.session_state.flashcards:
            difficulty = st.selectbox("Difficulty Filter:", ["All", "Easy", "Medium", "Hard"])
            
            if st.button("🔄 Shuffle Cards"):
                np.random.shuffle(st.session_state.flashcards)
                st.success("Cards shuffled!")
            
            if st.button("📊 Study Session"):
                start_flashcard_session()
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_flashcards_from_content():
    """Generate flashcards from the uploaded content"""
    if hasattr(st.session_state.studymate, 'chunks') and st.session_state.studymate.chunks:
        # Get text from chunks
        text_content = " ".join([chunk['text'] for chunk in st.session_state.studymate.chunks[:5]])
        
        # Generate flashcards
        new_flashcards = st.session_state.studymate.generate_flashcards(text_content, num_cards=8)
        
        if 'flashcards' not in st.session_state:
            st.session_state.flashcards = []
        
        st.session_state.flashcards.extend(new_flashcards)
        st.success(f"✨ Generated {len(new_flashcards)} new flashcards!")
    else:
        st.warning("Please upload a document first!")

def start_flashcard_session():
    """Start an interactive flashcard study session"""
    if st.session_state.flashcards:
        st.session_state.study_session_active = True
        st.session_state.current_card_index = 0
        st.session_state.session_score = 0
        st.success("🎯 Study session started! Good luck!")

def render_quiz_mode():
    """Render quiz creation and taking mode"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Quiz Arena")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not hasattr(st.session_state, 'quiz_questions'):
            st.session_state.quiz_questions = []
        
        tab1, tab2 = st.tabs(["📝 Create Quiz", "🎮 Take Quiz"])
        
        with tab1:
            if st.button("🎲 Generate Quiz Questions"):
                if hasattr(st.session_state.studymate, 'chunks') and st.session_state.studymate.chunks:
                    generate_quiz_from_content()
                else:
                    st.warning("Please upload a document first!")
            
            # Display generated questions
            if st.session_state.quiz_questions:
                st.markdown(f"### 📋 Generated Questions ({len(st.session_state.quiz_questions)})")
                
                for i, question in enumerate(st.session_state.quiz_questions):
                    with st.expander(f"Question {i+1}"):
                        st.write(f"*Q:* {question['question']}")
                        st.write("*Options:*")
                        for j, option in enumerate(question['options']):
                            st.write(f"{chr(65+j)}. {option}")
                        st.write(f"*Correct Answer:* {chr(65+question['correct_answer'])}")
        
        with tab2:
            if st.session_state.quiz_questions:
                render_interactive_quiz()
            else:
                st.info("Generate quiz questions first to start taking the quiz!")
    
    with col2:
        st.markdown("### 🏆 Quiz Stats")
        
        if hasattr(st.session_state, 'quiz_scores'):
            avg_score = np.mean(st.session_state.quiz_scores) if st.session_state.quiz_scores else 0
            st.metric("Average Score", f"{avg_score:.1f}%")
            st.metric("Quizzes Taken", len(st.session_state.quiz_scores))
        
        st.markdown("### 🎯 Difficulty Settings")
        difficulty = st.select_slider(
            "Quiz Difficulty:",
            options=["Easy", "Medium", "Hard", "Expert"],
            value="Medium"
        )
        
        num_questions = st.slider("Number of Questions:", 3, 15, 5)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_quiz_from_content():
    """Generate quiz questions from the uploaded content"""
    if hasattr(st.session_state.studymate, 'chunks') and st.session_state.studymate.chunks:
        text_content = " ".join([chunk['text'] for chunk in st.session_state.studymate.chunks[:3]])
        
        new_questions = st.session_state.studymate.generate_quiz(text_content, num_questions=6)
        
        if 'quiz_questions' not in st.session_state:
            st.session_state.quiz_questions = []
        
        st.session_state.quiz_questions.extend(new_questions)
        st.success(f"🎯 Generated {len(new_questions)} quiz questions!")

def render_interactive_quiz():
    """Render interactive quiz taking interface"""
    if 'current_quiz_index' not in st.session_state:
        st.session_state.current_quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_started = False
    
    if not st.session_state.quiz_started:
        if st.button("🚀 Start Quiz", type="primary"):
            st.session_state.quiz_started = True
            st.rerun()
    else:
        current_q = st.session_state.quiz_questions[st.session_state.current_quiz_index]
        
        st.markdown(f"### Question {st.session_state.current_quiz_index + 1} of {len(st.session_state.quiz_questions)}")
        st.write(current_q['question'])
        
        # Radio buttons for answers
        answer = st.radio(
            "Choose your answer:",
            current_q['options'],
            key=f"quiz_q_{st.session_state.current_quiz_index}"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.current_quiz_index > 0:
                if st.button("⬅ Previous"):
                    st.session_state.current_quiz_index -= 1
                    st.rerun()
        
        with col2:
            if st.button("✅ Submit Answer"):
                st.session_state.quiz_answers[st.session_state.current_quiz_index] = current_q['options'].index(answer)
                
                if st.session_state.current_quiz_index < len(st.session_state.quiz_questions) - 1:
                    st.session_state.current_quiz_index += 1
                    st.rerun()
                else:
                    # Quiz completed
                    show_quiz_results()
        
        with col3:
            if st.session_state.current_quiz_index < len(st.session_state.quiz_questions) - 1:
                if st.button("➡ Next"):
                    st.session_state.current_quiz_index += 1
                    st.rerun()

def show_quiz_results():
    """Show quiz results and score"""
    correct_answers = 0
    total_questions = len(st.session_state.quiz_questions)
    
    for i, question in enumerate(st.session_state.quiz_questions):
        if i in st.session_state.quiz_answers:
            if st.session_state.quiz_answers[i] == question['correct_answer']:
                correct_answers += 1
    
    score = (correct_answers / total_questions) * 100
    
    # Store score
    if 'quiz_scores' not in st.session_state:
        st.session_state.quiz_scores = []
    st.session_state.quiz_scores.append(score)
    
    # Display results
    st.balloons()
    st.markdown(f"""
    <div class="feature-card" style="text-align: center;">
        <h2>🎉 Quiz Complete! 🎉</h2>
        <h3>Your Score: {score:.1f}%</h3>
        <p>You got {correct_answers} out of {total_questions} questions correct!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset quiz
    if st.button("🔄 Take Another Quiz"):
        st.session_state.current_quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_started = False
        st.rerun()

def render_analytics_mode():
    """Render study analytics and progress tracking"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Study Analytics Dashboard")
    
    # Get analytics data
    analytics = st.session_state.studymate.get_study_analytics()
    
    if not analytics:
        st.info("📈 Start studying to see your analytics!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{analytics.get('total_sessions', 0)}</h3>
            <p>Study Sessions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        study_hours = analytics.get('total_study_time', 0) / 60
        st.markdown(f"""
        <div class="metric-card">
            <h3>{study_hours:.1f}h</h3>
            <p>Total Study Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_perf = analytics.get('avg_performance', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_perf:.1f}%</h3>
            <p>Avg Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        weekly_sessions = analytics.get('sessions_this_week', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{weekly_sessions}</h3>
            <p>This Week</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Study time over time
        if st.session_state.studymate.study_sessions:
            df = pd.DataFrame(st.session_state.studymate.study_sessions)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            daily_study = df.groupby('date')['duration'].sum().reset_index()
            
            fig = px.line(
                daily_study, 
                x='date', 
                y='duration',
                title="📈 Daily Study Time",
                color_discrete_sequence=['#006994']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance distribution
        if hasattr(st.session_state, 'quiz_scores') and st.session_state.quiz_scores:
            fig = px.histogram(
                x=st.session_state.quiz_scores,
                title="🎯 Quiz Score Distribution",
                color_discrete_sequence=['#4682B4']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Score (%)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Study streak and achievements
    st.markdown("### 🏆 Achievements & Streaks")
    
    achievements = [
        {"name": "First Steps", "desc": "Completed your first study session", "icon": "🎯"},
        {"name": "Flash Master", "desc": "Created 10+ flashcards", "icon": "📚"},
        {"name": "Quiz Ace", "desc": "Scored 90%+ on a quiz", "icon": "🏆"},
        {"name": "Consistent Learner", "desc": "Studied 3 days in a row", "icon": "🔥"},
    ]
    
    cols = st.columns(len(achievements))
    for i, achievement in enumerate(achievements):
        with cols[i]:
            # Check if achievement is unlocked (simplified logic)
            unlocked = i == 0 or (i == 1 and len(st.session_state.get('flashcards', [])) >= 10)
            
            st.markdown(f"""
            <div class="feature-card" style="text-align: center; opacity: {'1' if unlocked else '0.5'}">
                <div style="font-size: 2rem;">{achievement['icon']}</div>
                <h4>{achievement['name']}</h4>
                <p style="font-size: 0.9rem;">{achievement['desc']}</p>
                {'✅ Unlocked!' if unlocked else '🔒 Locked'}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_settings_mode():
    """Render settings and preferences"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings & Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Profile Settings")
        
        new_name = st.text_input("Your Name:", value=st.session_state.user_profile['name'])
        if new_name != st.session_state.user_profile['name']:
            st.session_state.user_profile['name'] = new_name
        
        study_goal = st.slider("Daily Study Goal (minutes):", 15, 240, 60)
        
        st.markdown("#### 🔔 Notifications")
        reminder_enabled = st.checkbox("Study Reminders", value=True)
        achievement_notifications = st.checkbox("Achievement Notifications", value=True)
        
        st.markdown("#### 🎨 Appearance")
        theme_options = ["Ocean Blue (Current)", "Forest Green", "Sunset Orange", "Night Mode"]
        selected_theme = st.selectbox("Theme:", theme_options)
        
    with col2:
        st.markdown("#### 📚 Study Preferences")
        
        default_flashcard_count = st.slider("Default Flashcards per Session:", 5, 20, 10)
        default_quiz_questions = st.slider("Default Quiz Questions:", 3, 15, 5)
        
        difficulty_preference = st.selectbox(
            "Preferred Difficulty:",
            ["Auto-adjust", "Easy", "Medium", "Hard"]
        )
        
        st.markdown("#### 💾 Data Management")
        
        col_export, col_clear = st.columns(2)
        
        with col_export:
            if st.button("📥 Export Data"):
                export_user_data()
        
        with col_clear:
            if st.button("🗑 Clear All Data", type="secondary"):
                if st.button("⚠ Confirm Clear", type="secondary"):
                    clear_all_user_data()
        
        st.markdown("#### ℹ About")
        st.info("""
        *StudyMate Pro v1.0*
        
        Your AI-powered study companion built with:
        - Streamlit for the interface
        - FAISS for semantic search
        - SentenceTransformers for embeddings
        - Plotly for analytics
        
        Made with 💙 for learners everywhere!
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def export_user_data():
    """Export user study data"""
    export_data = {
        'profile': st.session_state.user_profile,
        'chat_history': st.session_state.chat_history,
        'flashcards': st.session_state.get('flashcards', []),
        'quiz_questions': st.session_state.get('quiz_questions', []),
        'quiz_scores': st.session_state.get('quiz_scores', []),
        'study_sessions': st.session_state.studymate.study_sessions,
        'export_date': datetime.now().isoformat()
    }
    
    json_data = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="💾 Download Study Data",
        data=json_data,
        file_name=f"studymate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("📊 Data export ready!")

def clear_all_user_data():
    """Clear all user data"""
    keys_to_clear = [
        'chat_history', 'flashcards', 'quiz_questions', 'quiz_scores',
        'current_file_hash', 'uploaded_docs', 'study_session_active'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset studymate backend
    st.session_state.studymate = StudyMateProBackend()
    
    # Reset user profile
    st.session_state.user_profile = {
        'name': 'Student',
        'study_streak': 0,
        'total_study_time': 0,
        'achievements': []
    }
    
    st.success("🗑 All data cleared!")
    st.rerun()

def generate_study_summary():
    """Generate a study summary from recent activity"""
    if not st.session_state.chat_history:
        st.warning("No study activity to summarize yet!")
        return
    
    # Create a simple summary
    topics_covered = []
    questions_asked = len(st.session_state.chat_history)
    
    for chat in st.session_state.chat_history[-5:]:
        topic = chat['question'][:30]
        if topic not in topics_covered:
            topics_covered.append(topic)
    
    summary = f"""
    ## 📋 Study Session Summary
    
    *Questions Asked:* {questions_asked}
    
    *Topics Covered:*
    {chr(10).join([f"• {topic}..." for topic in topics_covered])}
    
    *Flashcards Created:* {len(st.session_state.get('flashcards', []))}
    
    *Quiz Questions Generated:* {len(st.session_state.get('quiz_questions', []))}
    
    *Keep up the great work! 🌟*
    """
    
    st.markdown(summary)

if __name__ == "__main__":
    main()