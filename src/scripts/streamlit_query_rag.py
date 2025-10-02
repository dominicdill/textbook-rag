import pandas as pd
import streamlit as st

from src.llm.rag import RAG
from src.db.db_manager import db_manager
from src.retrieval.query import Query
from src.settings import settings

# Set up the Streamlit page
st.set_page_config(page_title="RAG-Bot", layout="wide")
st.title("📚 Textbook RAG")
st.subheader("Answering questions from your documents")

@st.cache_resource
def init_rag():
    """Initialize the RAG model and cache it."""
    return RAG()

def display_sql_documents():
    """Fetches and displays documents from the database in a Streamlit table."""
    documents = db_manager.inspect_documents()
    if not documents:
        st.write("No documents to display.")
        return
    
    df = pd.DataFrame(documents)
    st.dataframe(df)

def display_retrieved_chunks(chunks):
    """Display information about retrieved chunks in a table."""
    if not chunks:
        st.write("No chunks retrieved.")
        return
    
    # Extract information from chunks
    chunk_data = []
    for i, chunk in enumerate(chunks):
        # Handle both chunk objects and serialized chunk dictionaries
        if isinstance(chunk, dict):
            # This is serialized chunk data from session state
            chunk_info = {
                'Filename': chunk.get('doc_name', 'Unknown'),
                'Page Start': chunk.get('page_start', 'N/A'),
                'Page End': chunk.get('page_end', 'N/A'),
                'Text': chunk.get('text', ''),#[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                'Score': chunk.get('score', 'N/A')
            }
        else:
            # This is an original chunk object
            filename = chunk.metadata.get('origin', {}).get('filename', 'Unknown')
            chunk_info = {
                'Filename': filename,
                'Page Start': chunk.page_start,
                'Page End': chunk.page_end,
                'Text': chunk.text,#[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'Score': chunk.score
            }
        chunk_data.append(chunk_info)


        # filename = chunk.metadata.get('origin', {}).get('filename', 'Unknown')
        # chunk_info = {
        #     'Filename': filename,
        #     'Page Start': chunk.page_start,
        #     'Page End': chunk.page_end,
        #     'Text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,  # Truncate long text
        #     'Score': chunk.score
        # }
        # chunk_data.append(chunk_info)
    
    # Create DataFrame and display
    df = pd.DataFrame(chunk_data)
    st.dataframe(df, width='stretch')


def set_hyperparmeters():
    # Place sliders in the sidebar
    with st.sidebar:
        st.header("Hyperparameter Tuning")

        # Use the settings object for default values and update it on change
        settings.retrieval_top_k = st.slider(
            "Retrieval Top K",
            min_value=1,
            max_value=20,
            value=settings.retrieval_top_k,
            help="The number of initial documents to retrieve."
        )

        settings.reranker_keep_top_q = st.slider(
            "Reranker Top K",
            min_value=1,
            max_value=20,
            value=settings.reranker_keep_top_q,
            help="The number of documents to keep after reranking."
        )

        settings.hybrid_retrieval_alpha = st.slider(
            "Hybrid Retrieval Alpha",
            min_value=0.0,
            max_value=1.0,
            value=settings.hybrid_retrieval_alpha,
            step=0.1,
            help="Weight for semantic search (1-alpha for keyword search)."
        )

        settings.query_expansion_number = st.slider(
            "Number of Expanded Queries",
            min_value=0,
            max_value=5,
            value=settings.query_expansion_number,
            help="How many alternative queries to generate from the original."
        )
    return None


def main():
    """Main function to run the Streamlit app."""
    set_hyperparmeters()
    rag = init_rag()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display chunks if they exist for this message
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("View Retrieved Chunks"):
                    display_retrieved_chunks(message["chunks"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                query = Query(text=prompt)
                
                # Pass the chat history to the generate method
                chat_history = st.session_state.get("messages", [])

                response, rel_chunks = rag.generate(query, chat_history=chat_history)

                # Store retrieved chunks in session state
                st.session_state.last_retrieved_chunks = rel_chunks

                    # Sidebar with a button to display documents
                # If there's a response, display it and add to history
                if response:
                    message_placeholder.markdown(response)
                    # st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Extract serializable chunk data
                    chunk_data = []
                    for chunk in rel_chunks:
                        chunk_info = {
                            'text': chunk.text,
                            'doc_name': chunk.metadata.get('origin', {}).get('filename', 'Unknown'),
                            'page_start': chunk.page_start,
                            'page_end': chunk.page_end,
                            'score': chunk.score
                        }
                        chunk_data.append(chunk_info)
                    
                    # Store both response and chunks in message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "chunks": chunk_data  # Store serialized chunk data
                    })
                    
                    # Display chunks in an expander
                    with st.expander("View Retrieved Chunks"):
                        display_retrieved_chunks(rel_chunks)
                else:
                    message_placeholder.markdown("No response generated.")

    with st.sidebar:
        if st.button("Inspect Documents"):
            display_sql_documents()

        if "last_retrieved_chunks" in st.session_state and st.session_state.last_retrieved_chunks:
            st.header("Retrieved Chunks")
            display_retrieved_chunks(st.session_state.last_retrieved_chunks)
                
if __name__ == "__main__":

       
    main()

    # Note: The db_manager.close() is not explicitly called here because
    # Streamlit's lifecycle can make resource cleanup tricky. The connection
    # pool should handle connections timing out. For a production app,
    # you might implement a more sophisticated cleanup mechanism.