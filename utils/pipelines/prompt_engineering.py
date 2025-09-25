

def initial_prompt(query):
    prompt = f'''SYSTEM: You extract search terms for a legal vector+BM25 hybrid retriever.

    Rules:
    - Return ONLY a JSON object. No prose.
    - Include: keywords (unigrams), key_phrases (multi-word), entities (case names, statutes), negatives (stop-words to ignore), expansions (common synonyms/abbrevs).
    - Preserve exact legal phrases (e.g., “reasonable suspicion”, “stop and frisk”).
    - Use lowercase except proper names/citations.
    - No hallucinations; don’t invent case names/citations.

    USER QUERY: "{query}"

    OUTPUT JSON SCHEMA:
    {{
    "keywords": [string],
    "key_phrases": [string],
    "entities": [string],
    "expansions": [string],
    "negatives": [string]
    }}'''
    return prompt

def summarize_opinion_prompt(opinion, user_query):
    
    prompt = f"""
    You are a legal assistant. Read the following court opinion and extract the most important key phrases,
    concepts, and terms that summarize what the opinion is about. Keep the output concise and easy to scan.

    User Query:
    {user_query}
    
    Court Opinion:
    {opinion}

    Key Phrases:
    """
    return prompt