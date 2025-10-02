from loguru import logger

from src.documents.pdf import Chunk
from src.retrieval.query import Query
from src.retrieval.retrieval import Retriever
from src.llm.client import llm_connection
from src.settings import settings

class RAG:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = llm_connection()

    def generate(self, query: Query, top_k: int = settings.retrieval_top_k, chat_history: list = []) -> tuple[str|None, list[Chunk]]:
        logger.info("Starting RAG generation process...")
        rel_chunks = self.retriever.retrieve_with_expansion(query, top_k=top_k)
        logger.info(f"Retrieved {len(rel_chunks)} relevant chunks.")

        original_query = query.text
        context = ""
        for i, chunk in enumerate(rel_chunks):
            chunk_metadata = chunk.metadata
            source = chunk_metadata['origin']['filename']
            page_start = chunk.page_start
            page_end = chunk.page_end
            logger.info(f"Chunk {i+1}: Source: {source}, Pages: {page_start}-{page_end}")
            logger.info(f"Chunk {i+1} Text: {chunk.text}")  # Log first 200 characters
            context += f"\nSource: {source} pages {page_start}-{page_end}:\n{chunk.text}\n"
            
        # prompt = f"Use the following context to aid you in answering the question:\n\nContext:\n{context}\n\nQuestion: {original_query}\nAnswer:"

        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        prompt = (
            f"<ConversationHistory>\n{history_context}\n</ConversationHistory>\n"
            f"<Context>\n{context}\n</Context>\n"
            f"<Question> {query.text}\n</Question>\n"
            "Answer:"
        )

        self.llm.set_prompt(prompt)

        # system_message = (
        #     "You are an expert assistant designed to provide accurate and concise answers "
        #     "to user questions. Use the information in the <Context> "
        #     "section to inform your response. If the context does not contain relevant information, "
        #     "tell the user, but still be as helpful as possible. "
        #     "The user is not aware of the context provided, so always answer in a way that is clear and complete."
        # )

            # A more flexible and powerful system message
        system_message = (
            "You are an expert assistant with access to a specialized knowledge base. "
            "Your primary goal is to provide accurate and helpful answers to user questions. "
            "Use the information provided in the <Context> section to form the foundation of your response. "
            "If the context provides a direct answer, use it. If the context is only partially helpful "
            "or lacks the necessary detail, you are encouraged to synthesize the information from the context "
            "with your own general knowledge to provide a comprehensive and complete answer. "
            "If the context is not relevant at all, rely on your general knowledge but inform the user "
            "that the provided documents did not contain the information. "
            "Additionally, consider the entire conversation history in the <ConversationHistory> section if it is relevant."
            "Always answer in a clear, complete, and helpful manner."
            "Reference the sources of your information by mentioning the document names and page numbers when applicable."
        )
        self.llm.set_system_message(system_message)
        response = self.llm.generate_prompt_response()
        return response, rel_chunks