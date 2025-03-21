import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Tuple
import gc

class EmbeddingsManager:
    def __init__(self, use_sklearn=True):
        """Initialize with sklearn-based vectorizer to avoid memory issues"""
        self.use_sklearn = use_sklearn
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.matrix = None
        
    def chunk_text(self, df: pd.DataFrame, chunk_size=200, overlap=20) -> List[Dict]:
        """Break text into overlapping chunks"""
        chunks = []
        
        for idx, row in df.iterrows():
            text = row['Text']
            if not isinstance(text, str):
                continue
                
            words = text.split()
            if len(words) < chunk_size:
                # Document is smaller than chunk size, keep as is
                chunk = {
                    'text': text,
                    'source': row.get('Source', f"Document {idx}"),
                    'document_id': idx
                }
                chunks.append(chunk)
                continue
            
            # Create overlapping chunks
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text.strip()) > 0:
                    chunk = {
                        'text': chunk_text,
                        'source': row.get('Source', f"Document {idx}"),
                        'document_id': idx
                    }
                    chunks.append(chunk)
        
        self.chunks = chunks
        return chunks

    def create_index(self, chunks: List[Dict]) -> None:
        """Create TF-IDF vectors from text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        # Force garbage collection to free memory
        gc.collect()
    
    def process_dataframe(self, df: pd.DataFrame, chunk_size=200, overlap=20) -> None:
        """Process a dataframe to create chunks and index"""
        chunks = self.chunk_text(df, chunk_size, overlap)
        self.create_index(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for the k most similar chunks to the query"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['score'] = float(similarities[idx])
                results.append(result)
                
        return results

    def get_relevant_context(self, query: str, max_chunks=5) -> Tuple[str, List[str]]:
        """Get relevant context for a query and return sources"""
        results = self.search(query, k=max_chunks)
        context = "\n\n".join([r['text'] for r in results])
        sources = [r['source'] for r in results]
        return context, sources
