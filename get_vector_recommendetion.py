from rephrase_query import QueryRephraser
from bulid_vec_db import VectorStoreManager
from typing import List, Dict, Union
import streamlit as st

class recommendations_based_on_vecdb:
    """
    Class to handle recommendations based on vector database.
    """
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = vector_store_path
        self.vector_manager = VectorStoreManager()
        self.vec_db = self.vector_manager.load_vector_store(self.vector_store_path)

    def get_vector_recommendations(self, user_query: str, k: int = 30) -> Union[str, List[Dict]]:
        """
        Get recommendations based on user intent using vector similarity search.
        
        Args:
            user_query (str): The user's query or intent
            vec_db (FAISS): The FAISS vector store containing the items
            k (int): Number of recommendations to return (default: 20)
            
        Returns:
            Union[str, List[Dict]]: user_intent,  List of recommended items with their metadata and scores
        """
        
        query_rephraser = QueryRephraser()
        user_intent = query_rephraser.rephrase_query(user_query)

        try:

            relevant_items = self.vec_db.similarity_search(
                user_intent,
                k=k
            )
            
            recommended_items = []
            for item in relevant_items:
            
                recommendation = {
                    'id': item.metadata.get('image_index', 'N/A'),
                    'content': item.page_content,
                                            }
                recommended_items.append(recommendation)
                
            return user_intent, recommended_items
            
        except Exception as e:
            st.error(f"Error during getting vector_recommendations: {str(e)}")
