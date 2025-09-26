

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

def summarize_irac_prompt(opinion, prompt):
    SYSTEM_PROMPT = (
        "You are ACE, an Attorney Case Expert. Your job is to summarize court opinions using only the "
        "provided context. Do not invent facts, rules, holdings, or citations.\n\n"
        "Follow this exact structure:\n"
        "Key Holdings:\n"
        "- List 2–4 short bullet points of the court’s main holdings, quoting or paraphrasing directly.\n\n"
        "Facts:\n"
        "- Concisely restate the relevant background facts from the opinion.\n\n"
        "Issue:\n"
        "- Frame the precise legal question before the court.\n\n"
        "Rule:\n"
        "- State the controlling rule(s) of law, citing directly to the opinion.\n\n"
        "Application:\n"
        "- Explain how the court applied the rule to the facts. Avoid generic phrases like "
        "'balancing interests' unless the opinion uses them.\n\n"
        "Conclusion:\n"
        "- State the outcome of the case in one sentence.\n\n"
        "References:\n"
        "- Include exact citations (case names, reporters, or URLs) found in the text. "
        "Preserve URLs exactly as written. Do not fabricate references.\n\n"
        "If a section is not supported by the text, write 'Not stated in opinion.'"
    )
    prompt = f"""
        {SYSTEM_PROMPT}
        User Query: {prompt}
        Court Opinion: {opinion}
    """
    return prompt