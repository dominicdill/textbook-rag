import json
from typing import List

from loguru import logger

from src.embeddings.openai_embedding import embed_texts
from src.llm.client import llm_connection
from src.settings import settings

class Query:
    def __init__(self, text: str):
        self.text: str = text
    
    def embed(self):
        self.embedding = embed_texts([self.text])[0]
    

class QueryExpansionService:
    """Placeholder service for LLM-based query expansion"""
    
    def expand_query(self, original_query: Query) -> List[str]:
        """
        Use an LLM to generate expanded versions of the original query.
        Uses structured output to ensure consistent formatting.
        """
        content = f" \
        You are an expert at creating search queries for information retrieval.  \
        Information will be retrieved using both semantic and keyword-based search.  \
        Given an initial query, generate {settings.query_expansion_number} expanded versions of the query.  \
        Expanded queries should be crafted in such a way that they will retrieve diverse and relevant results. \
        If the original query is long, break it down into shorter, more focused queries.  \
        Each expanded query should be a concise phrase or question that captures a different aspect of the original query.  \
        Avoid redundancy between the expanded queries.  \
        The query to expand is shown below.\n \
        Query: '{original_query.text}'"


        messages = [
            {
            "role": "system", 
            "content": content
            }
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
            "name": "query_expansions",
            "strict": True, 
            "schema": {
                "type": "object",
                "properties": {
                "expansions": {
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "An expanded version of the original query"
                    },
                    "minItems": settings.query_expansion_number,
                    "maxItems": settings.query_expansion_number
                }
                },
                "required": ["expansions"],
                "additionalProperties": False
            }
            }
        }
        expanded_queries = []
        try:
            
            llm = llm_connection()
            response = llm.generate_formatted_response(messages, response_format)
            
            response = response.model_dump()["choices"][0]['message']['content']
            response = json.loads(response)
            for item in response['expansions']:
                expanded_queries.append(item)
            logger.info(f"Generated {len(expanded_queries)} expanded queries.")
        except:
            logger.error("Error generating expanded queries.")
            logger.error("Falling back to original query only.")
            expanded_queries = []
        return expanded_queries


class QueryExpander:
    """Factory for creating expanded queries from an original query"""

    def __init__(self, expansion_service: QueryExpansionService = QueryExpansionService()):
        self.expansion_service = expansion_service

    def generate(self, original_query: Query) -> List[Query]:
        """
        Use the expansion service to generate expanded queries.
        
        Args:
            original_query: The original user query as a Query object
            
        Returns:
            List of Query objects including original and expanded variants
        """
        # Get expanded query texts from the LLM service
        expanded_query_texts = self.expansion_service.expand_query(original_query)
        
        # Convert to Query objects
        queries = [original_query]
        for query_text in expanded_query_texts:
            logger.info(f"Expanded query: {query_text}")
            query = Query(query_text)
            query.embed()
            queries.append(query)
        logger.info("Embedded all expanded queries.")
        return queries
    
