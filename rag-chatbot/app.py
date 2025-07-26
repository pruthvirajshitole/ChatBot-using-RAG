import streamlit as st
from utils.embedder import get_embedding
from utils.llm_utils import query_llm
import numpy as np

st.title("💬 RAG Chatbot")

# Add custom CSS for link-style button
st.markdown("""
    <style>
    .link-button {
        background: none;
        border: none;
        color: #6c757d;
        text-decoration: underline;
        cursor: pointer;
        font-size: 1rem;
        padding: 0;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for context and chat history
if "context_set" not in st.session_state:
    st.session_state.context_set = False
    st.session_state.context_text = ""
    st.session_state.context_chunks = []
    st.session_state.context_embeddings = []
    st.session_state.chat_history = []

# Button to change context at any time
if st.session_state.context_set:
    if st.button("Change Context"):
        st.session_state.context_set = False
        st.session_state.context_text = ""
        st.session_state.context_chunks = []
        st.session_state.context_embeddings = []
        st.session_state.latest_answer = ""
        st.session_state.clear_input = True
        st.rerun()

# Context input step
if not st.session_state.context_set:
    with st.form("context_form"):
        st.markdown('Enter context for this chat (optional):', unsafe_allow_html=True)
        context_text = st.text_area(" ", key="context_text_area")
        submit_context = st.form_submit_button("Set Context")
    if submit_context:
        st.session_state.context_text = context_text
        if context_text.strip():
            # Split context into chunks (paragraphs)
            context_chunks = [chunk.strip() for chunk in context_text.split('\n\n') if chunk.strip()]
            st.session_state.context_chunks = context_chunks
            st.session_state.context_embeddings = [get_embedding(chunk) for chunk in context_chunks]
        else:
            st.session_state.context_chunks = []
            st.session_state.context_embeddings = []
        st.session_state.context_set = True
        st.session_state.latest_answer = ""
        st.session_state.latest_question = ""
        st.session_state.clear_input = True
        st.session_state.show_input_box = True
        st.rerun()
    st.stop()

# Initialize the flag
if "show_input_box" not in st.session_state:
    st.session_state.show_input_box = True
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""

# Track the latest question
if "latest_question" not in st.session_state:
    st.session_state.latest_question = ""

# After context is set
if st.session_state.context_set:
    # Show question and answer if they exist
    if st.session_state.latest_answer and st.session_state.latest_question:
        st.write("**You asked:**")
        st.write(st.session_state.latest_question)
        st.write("🧠 **Answer:**")
        st.write(st.session_state.latest_answer)
        # Show 'Ask Anything' button below the answer
        if st.button("Ask Anything"):
            st.session_state.show_input_box = True
            st.session_state.latest_answer = ""
            st.session_state.latest_question = ""
            st.session_state.clear_input = True
            st.rerun()
        st.session_state.show_input_box = False
    # Show input box if flag is set and no answer is present
    elif st.session_state.show_input_box:
        user_input = st.text_input("Ask anything:", key="chat_input")
        if st.button("Send"):
            try:
                query_embedding = get_embedding(user_input)
                relevant_context = ""
                if st.session_state.context_embeddings:
                    def cosine_sim(a, b):
                        a = np.array(a)
                        b = np.array(b)
                        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    sims = [cosine_sim(query_embedding, emb) for emb in st.session_state.context_embeddings]
                    top_k = 3
                    top_indices = np.argsort(sims)[-top_k:][::-1]
                    relevant_chunks = [st.session_state.context_chunks[i] for i in top_indices if sims[i] > 0]
                    if relevant_chunks:
                        relevant_context = "\n".join(relevant_chunks)
                if relevant_context:
                    prompt = f"Answer the question based on the context below:\n\nContext:\n{relevant_context}\n\nQuestion: {user_input}"
                else:
                    prompt = user_input
                response = query_llm(prompt, provider="groq")
                if 'error' in response:
                    st.error(f"Error: {response['error']}")
                else:
                    if 'choices' in response:
                        answer = response['choices'][0]['message']['content']
                        st.session_state.latest_answer = answer
                        st.session_state.latest_question = user_input
                    else:
                        st.session_state.latest_answer = str(response)
                        st.session_state.latest_question = user_input
                st.session_state.clear_input = True
                st.session_state.show_input_box = False
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API keys and make sure they are valid.")
    else:
        # If neither input nor answer, show nothing (shouldn't happen)
        pass

# After context change or set, also reset show_input_box
# (add this where you clear context)
st.session_state.show_input_box = True
st.session_state.latest_answer = ""
st.session_state.latest_question = ""