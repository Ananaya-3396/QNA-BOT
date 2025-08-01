import streamlit as st
import pandas as pd
import torch
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page title and configuration
st.set_page_config(page_title="HR Q&A Assistant", layout="wide")

# Load Q&A CSV
@st.cache_data()
def load_hr_qa():
    try:
        df = pd.read_csv("HR questions.csv", encoding="cp1252")
        questions = df['input'].tolist()
        answers = df['output'].tolist()
        return df, questions, answers
    except Exception as e:
        st.error(f"Error loading Q&A data: {str(e)}")
        return pd.DataFrame(), [], []

# Load sentence transformer model for semantic search
@st.cache_resource()
def load_embedding_model():
    try:
        # Using a smaller, faster model suitable for semantic search
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# Generate embeddings for all questions
@st.cache_data(show_spinner=True)
def generate_embeddings(_model, questions):
    try:
        if _model is None:
            return None
        embeddings = _model.encode(questions)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Find best match using multiple methods
def get_best_match(user_q, questions, answers, embeddings=None, model=None, threshold=0.7):
    results = []
    
    # Method 1: Fuzzy string matching
    try:
        fuzzy_match, fuzzy_score, fuzzy_idx = process.extractOne(user_q, questions, scorer=fuzz.token_set_ratio)
        fuzzy_score_normalized = fuzzy_score / 100  # Normalize to 0-1 range
        results.append((fuzzy_idx, fuzzy_score_normalized, "fuzzy"))
    except Exception as e:
        st.error(f"Error in fuzzy matching: {str(e)}")
    
    # Method 2: Semantic similarity with embeddings
    if embeddings is not None and model is not None:
        try:
            # Generate embedding for user question
            user_embedding = model.encode([user_q])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(user_embedding, embeddings)[0]
            
            # Get top match
            semantic_idx = np.argmax(similarities)
            semantic_score = similarities[semantic_idx]
            
            results.append((semantic_idx, semantic_score, "semantic"))
        except Exception as e:
            st.error(f"Error in semantic matching: {str(e)}")
    
    # Choose best result
    if results:
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score, method = results[0]
        
        if best_score >= threshold:
            return answers[best_idx], questions[best_idx], best_score, method
    
    return None, "", 0, ""

# Main application
def main():
    # Title and description
    st.title("HR Q&A Assistant")
    st.write("Ask me anything about my professional experience, and my bot will answer just like I would.")
    
    # Load data
    df, questions, answers = load_hr_qa()
    
    # Load embedding model
    model = load_embedding_model()
    
    # Generate embeddings
    embeddings = generate_embeddings(model, questions)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # User input
    user_input = st.text_input("Enter your question please:")
    
    # Get answer button
    if st.button("Get Answer") and user_input:
        answer, matched_q, score, method = get_best_match(
            user_input, questions, answers, embeddings, model
        )
        
        if answer:
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": answer,
                "matched_question": matched_q,
                "score": score,
                "method": method
            })
        else:
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": "I don't have enough information to answer that question. Please try asking something else.",
                "matched_question": "",
                "score": 0,
                "method": ""
            })
    
    # Display current answer
    if st.session_state['chat_history']:
        latest = st.session_state['chat_history'][-1]
        st.markdown(f"**Answer:** {latest['answer']}")
    
    # Chat history in sidebar
    if st.session_state['chat_history']:
        st.sidebar.subheader("Chat History")
        for item in reversed(st.session_state['chat_history']):
            st.sidebar.markdown(f"**Question:** {item['question']}")
            st.sidebar.markdown(f"**Answer:** {item['answer']}")
    
    # Removed explanation section as requested

if __name__ == "__main__":

    main()
