"""
Simple fallback model that can be used when API calls fail
"""

def simple_keyword_response(query, context):
    """A basic keyword-based response generator as fallback"""
    query = query.lower()
    
    # Extract a relevant passage from the context based on keyword matching
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        for word in query.split():
            if word.lower() in sentence.lower() and len(word) > 3:  # Only match on meaningful words
                relevant_sentences.append(sentence)
                break
    
    if relevant_sentences:
        return "Based on the available information: " + ". ".join(relevant_sentences[:3]) + "."
    else:
        return "I couldn't find specific information about that in the available context."
