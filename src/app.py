"""
GitLab RAG Chatbot - Streamlit Web Interface (Production Version)

This uses the new production-grade package structure.
Run with: streamlit run src/app.py
"""

import os
import sys
import streamlit as st
import csv
import datetime
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from new production-grade package
from gitlab_rag_chatbot.config.settings import settings
from gitlab_rag_chatbot.core.retrieval.vector_store import DocumentRetriever
from gitlab_rag_chatbot.core.embeddings.providers import get_embedding_function
from gitlab_rag_chatbot.web.feedback_collector import feedback_collector
from gitlab_rag_chatbot.utils.logging_setup import setup_application_logging

# Import chat function from migrated structure
from src.model_providers.chat import get_chat_fn

# Setup logging
setup_application_logging(log_level="INFO")

st.set_page_config(
    page_title="GitLab Handbook Chatbot", 
    page_icon="💬", 
    layout="wide"
)

# Initialize components with caching
@st.cache_resource(show_spinner=False)
def initialize_components():
    """Initialize RAG components with caching for performance."""
    embedding_function = get_embedding_function()
    chat_function = get_chat_fn()
    document_retriever = DocumentRetriever()
    return embedding_function, chat_function, document_retriever

embedding_function, chat_function, document_retriever = initialize_components()

# UI Layout
st.title("🤖 GitLab Handbook & Direction Chatbot")
st.caption("Production-grade RAG system for GitLab documentation with semantic search and AI responses.")

# Initialize chat history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar with information
with st.sidebar:
    st.header("📊 System Information")
    st.markdown("**Architecture**: Production-grade RAG pipeline")
    st.markdown("**Scope**: GitLab Handbook & Direction docs")
    st.markdown("**Features**: Semantic search, source citations, user feedback")
    
    st.divider()
    
    # Display system stats
    collection_stats = document_retriever.get_collection_stats()
    st.markdown(f"**Documents**: {collection_stats.get('total_documents', 0):,}")
    st.markdown(f"**Vector Store**: `{settings.vector_store_directory}`")
    
    st.divider()
    
    # Configuration display
    st.markdown("**Configuration**")
    st.markdown(f"- Provider: `{settings.embedding_provider}`")
    st.markdown(f"- Top-K Results: `{settings.top_k_results}`")
    st.markdown(f"- Min Similarity: `{settings.minimum_similarity_score}`")
    
    st.divider()
    
    # Conversation context status
    st.markdown("**Conversation**")
    conversation_length = len(st.session_state.conversation_history)
    if conversation_length == 0:
        st.markdown("🆕 New conversation")
    else:
        st.markdown(f"💬 {conversation_length // 2} exchanges")
        if conversation_length > 1:
            st.markdown("✨ Context-aware mode active")

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Ask about GitLab's Handbook or Direction...")

if user_question:
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user", 
        "content": user_question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Show different spinner message for follow-up questions
        spinner_message = "🔍 Searching documentation and generating response..."
        if len(st.session_state.conversation_history) > 1:
            spinner_message = "🔍 Analyzing conversation context and searching documentation..."
        
        with st.spinner(spinner_message):
            
            # Create context-aware search query
            search_query = user_question
            conversation_context = ""
            
            # If we have conversation history, create enhanced search query for follow-up questions
            if len(st.session_state.conversation_history) > 1:
                # Get recent conversation context (last 2-3 exchanges)
                recent_history = st.session_state.conversation_history[-6:]  # Last 3 Q&A pairs
                
                # Build conversation context for better search
                context_parts = []
                for msg in recent_history:
                    if msg["role"] == "user":
                        context_parts.append(f"Previous question: {msg['content']}")
                    elif msg["role"] == "assistant":
                        # Extract key entities/topics from assistant response
                        response_preview = msg['content'][:200]  # First 200 chars
                        context_parts.append(f"Previous answer: {response_preview}")
                
                conversation_context = " | ".join(context_parts[-4:])  # Last 2 Q&A pairs
                
                # Create enhanced search query that combines current question with context
                search_query = f"{user_question} {conversation_context}"
            
            # Generate query embedding
            query_embedding = embedding_function([search_query])[0]
            
            # Search for similar documents
            search_results = document_retriever.search_similar_documents(
                query_embedding=query_embedding,
                max_results=settings.top_k_results
            )
            
            # Extract results
            documents = search_results.get("documents", [[]])[0]
            metadata_list = search_results.get("metadatas", [[]])[0]
            distances = search_results.get("distances", [[]])[0]
            
            # Convert distances to similarity scores (1 - distance)
            similarity_scores = [1 - (distance or 0) for distance in distances]
            
            # Prepare context and sources
            context_blocks = []
            source_urls = []
            
            for document, metadata, similarity in zip(documents, metadata_list, similarity_scores):
                source_url = metadata.get("url", "Unknown source")
                # Use the full chunk since we already did intelligent chunking during ingestion
                document_snippet = document
                context_blocks.append(f"[Source: {source_url}]\n{document_snippet}")
                source_urls.append(source_url)
            
            # Check if we have confident results
            max_similarity = max(similarity_scores) if similarity_scores else 0
            
            if max_similarity < settings.minimum_similarity_score:
                # Low confidence response
                response_message = (
                    "🤔 I couldn't find confident information about this in GitLab's documentation. "
                    "Try rephrasing your question or check these potentially related sources:\n\n"
                )
                for i, url in enumerate(source_urls[:3], 1):
                    response_message += f"{i}. {url}\n"
                
                st.markdown(response_message)
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": response_message
                })
                
            else:
                # Generate AI response with context
                context_text = "\n\n".join(context_blocks)
                
                # Build conversation-aware prompt
                conversation_history_text = ""
                if len(st.session_state.conversation_history) > 1:
                    # Include recent conversation for context (excluding current question)
                    recent_conversation = st.session_state.conversation_history[-4:-1]  # Last 2 Q&A pairs
                    history_parts = []
                    for msg in recent_conversation:
                        role_label = "Human" if msg["role"] == "user" else "Assistant"
                        history_parts.append(f"{role_label}: {msg['content']}")
                    
                    if history_parts:
                        conversation_history_text = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(history_parts)
                
                prompt = (
                    f"SYSTEM: You are a helpful assistant that answers questions using the provided context "
                    f"from GitLab's documentation. You can reference previous conversation to understand "
                    f"follow-up questions (like 'tell me more about her/him/it'). If the information isn't "
                    f"in the context, say so clearly.{conversation_history_text}\n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"USER QUESTION: {user_question}\n\n"
                    f"ASSISTANT:"
                )
                
                ai_response = chat_function([{"role": "user", "content": prompt}], temperature=0.2)
                
                # Display response
                st.markdown(ai_response)
                
                # Display sources
                with st.expander("📚 Sources & Similarity Scores"):
                    for i, (url, similarity) in enumerate(zip(source_urls, similarity_scores), 1):
                        confidence_emoji = "🟢" if similarity > 0.85 else "🟡" if similarity > 0.75 else "🔴"
                        st.markdown(f"{confidence_emoji} **{i}.** [{url}]({url}) — Similarity: {similarity:.3f}")
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": ai_response
                })
                
                # Feedback collection UI
                st.divider()
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 4])
                
                with feedback_col1:
                    if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.conversation_history)}"):
                        feedback_collector.record_feedback(
                            user_question=user_question,
                            chatbot_response=ai_response,
                            feedback_rating="helpful",
                            source_urls=source_urls,
                            similarity_scores=similarity_scores
                        )
                        st.toast("✅ Thank you for your feedback!")
                
                with feedback_col2:
                    if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.conversation_history)}"):
                        feedback_collector.record_feedback(
                            user_question=user_question,
                            chatbot_response=ai_response,
                            feedback_rating="not_helpful",
                            source_urls=source_urls,
                            similarity_scores=similarity_scores
                        )
                        st.toast("📝 Feedback logged. We'll work on improving!")

# Footer with system information
st.divider()
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🏗️ **Architecture**: Production-grade RAG")
    
    with col2:
        st.caption(f"🤖 **Provider**: {settings.embedding_provider.title()}")
    
    with col3:
        feedback_stats = feedback_collector.get_feedback_statistics()
        total_feedback = feedback_stats.get("total_feedback", 0)
        st.caption(f"📊 **Feedback**: {total_feedback} responses collected")
