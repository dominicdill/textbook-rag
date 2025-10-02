from rich import print
from src.llm.rag import RAG
from src.db.db_manager import db_manager
from src.retrieval.query import Query

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

console = Console()

def display_sql_documents():
    documents = db_manager.inspect_documents()

    if not documents:
        print("No documents to display.")
        return
        

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    
    column_configs = {
        "doc_name": {"ratio": 2, "no_wrap": True},
        "source_path": {"ratio": 1}, # Give source_path more space
        "page_count": {"justify": "right", "width": 5},
        "created_at": {"width": 10},
        "ingested_at": {"width": 10}
    }
    
    for column_name in documents[0].keys():
        config = column_configs.get(column_name, {})
        table.add_column(column_name, **config)
    
    for doc in documents:
        table.add_row(*[str(value) if value is not None else "" for value in doc.values()])

    console.print(table)


def main():
    console = Console()
    chat_history = []
    rag = RAG() # Initialize RAG once

    while True:
        query_text = input("Enter your question (or 'quit' to exit): ")
        if query_text.lower() == 'quit' or not query_text.strip():
            break

        query = Query(text=query_text)
        response = rag.generate(query, chat_history=chat_history)
        console.print("\n\n[bold green]Response:[/bold green]")
        if response:
            # Store the current exchange in history
            chat_history.append({"role": "user", "content": query_text})
            chat_history.append({"role": "assistant", "content": response})

            console.print(Markdown(response)) # Use Markdown for better display
        else:
            print("No response generated.")
        print("\n\n")
    
    db_manager.close()




if __name__ == "__main__":
    #display_sql_documents()
    main()
