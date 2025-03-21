import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class SimpleRAG:
    """A simple and memory-efficient RAG implementation"""
    
    def __init__(self, dataframe=None):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.df = dataframe
        self.matrix = None
        
    def load_data(self, dataframe):
        """Load data from dataframe"""
        self.df = dataframe
        
        # Create vectorizer matrix
        valid_texts = self.df['Text'].fillna('').astype(str).tolist()
        self.matrix = self.vectorizer.fit_transform(valid_texts)
        
    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        """Search for most relevant documents"""
        if self.matrix is None or self.df is None:
            return pd.DataFrame()
            
        # Transform query and get similarities
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        
        # Create results with scores
        result_df = self.df.copy()
        result_df['similarity'] = similarities
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity', ascending=False).head(k)
        return result_df
    
    def get_relevant_context(self, query: str, max_docs=3) -> Tuple[str, List[str]]:
        """Get relevant context and sources"""
        results = self.search(query, k=max_docs)
        
        if results.empty:
            return "No relevant information found.", []
        
        context = "\n\n".join(results['Text'].astype(str).tolist())
        sources = results.get('Source', results.index).tolist()
        
        return context, sources
