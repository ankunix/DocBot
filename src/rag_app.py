import pandas as pd
import numpy as np
import gc
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGApp:
    """
    Consolidated class for chunking, indexing, and retrieving context.
    Uses a TF-IDF approach for searching relevant chunks quickly.
    """

    def __init__(self,
                 chunk_size: int = 200,
                 overlap: int = 20,
                 max_docs: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_docs = max_docs

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []     # where chunks & metadata are stored
        self.matrix = None   # TF-IDF matrix of chunks

    def chunk_text(self, df: pd.DataFrame) -> List[Dict]:
        """
        Breaks input dataframe text into overlapping chunks.
        """
        if 'Text' not in df.columns:
            return []
        
        new_chunks = []
        for idx, row in df.iterrows():
            text = row['Text']
            if not isinstance(text, str):
                continue

            words = text.split()
            # If smaller than chunk_size, keep as-is
            if len(words) < self.chunk_size:
                new_chunks.append({
                    'text': text,
                    'source': row.get('Source', f"Document {idx}"),
                    'document_id': idx
                })
                continue

            # Create overlapping chunks
            step = self.chunk_size - self.overlap
            for i in range(0, len(words), step):
                chunk_text = ' '.join(words[i:i + self.chunk_size]).strip()
                if chunk_text:
                    new_chunks.append({
                        'text': chunk_text,
                        'source': row.get('Source', f"Document {idx}"),
                        'document_id': idx
                    })

        return new_chunks

    def create_index(self, chunk_list: List[Dict]) -> None:
        """Creates TF-IDF vectors from text chunks."""
        if not chunk_list:
            return

        texts = [ch['text'] for ch in chunk_list]
        self.matrix = self.vectorizer.fit_transform(texts)
        self.chunks = chunk_list
        gc.collect()  # force garbage collection to free memory

    def load_data(self, df: pd.DataFrame) -> None:
        """
        High-level method to chunk a dataframe and create an index.
        """
        chunked = self.chunk_text(df)
        self.create_index(chunked)

    def search_chunks(self, query: str, k: int = None) -> List[Dict]:
        """
        Search for k most similar chunks to the query.
        """
        if self.matrix is None or not self.chunks:
            return []

        if k is None:
            k = self.max_docs

        # Vectorize query and compute similarity
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]

        # Sort chunks by descending similarity
        top_indices = sims.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            chunk_copy = self.chunks[idx].copy()
            chunk_copy['score'] = float(sims[idx])
            results.append(chunk_copy)

        return results

    def get_relevant_context(self, query: str, max_chunks: int = None) -> Tuple[str, List[str]]:
        """
        Returns relevant extracted text and sources for a given query.
        """
        if max_chunks is None:
            max_chunks = self.max_docs

        top_chunks = self.search_chunks(query, k=max_chunks)
        if not top_chunks:
            return "No relevant information found.", []

        context = "\n\n".join(ch['text'] for ch in top_chunks)
        sources = [ch['source'] for ch in top_chunks]
        return context, sources
