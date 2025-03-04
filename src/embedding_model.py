"""
Embedding-based retrieval model
"""
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingRetrieval:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast model
        
    def preprocess_context(self, context):
        # Split into manageable chunks
        chunks = [context[i:i+512] for i in range(0, len(context), 256)]  # Overlapping chunks
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks)
        
    def answer_query(self, query):
        # Get query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Find most similar chunks
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 most similar chunks
        
        # Return top chunks as response
        response = "Based on the retrieved information:\n\n"
        for idx in top_indices:
            response += f"- {self.chunks[idx]}\n\n"
            
        return response

    def chat_completion(self, messages):
        # Extract query and context
        context = ""
        query = ""
        
        for msg in messages:
            if msg["role"] == "system":
                context = msg["content"].replace("You are an assistant that answers questions based on the following context: ", "")
            elif msg["role"] == "user":
                query = msg["content"]
        
        self.preprocess_context(context)
        response = self.answer_query(query)
        
        return {
            "choices": [
                {
                    "message": {
                        "content": response,
                        "role": "assistant"
                    }
                }
            ]
        }
