# prompts.py

FILTER_PREFERENCES_PROMPT = """
You are given:
- A user's personal background
- A conversation history
- A clarified user question

Your task is to extract **personal interests or preferences** from the user's background that may be **relevant to this search intent**.

Do not include general personal facts (e.g., age, height, appearance).
Only keep things that could affect what they are looking for (e.g., art interests, cooking, hiking, music, environmentalism).

Return a brief list of preferences in bullet point form.

# User Background:
{ptkb}

# Conversation History:
{history}

# Clarified User Question:
{question}

# Output Format Example:
- Interested in art and watercolor painting
- Enjoys cooking, especially Italian cuisine
- Likes hiking and nature activities
"""

QD_PROMPT = """
You are given:
- A set of **user background facts**
- A **conversation history** between a user and a system
- A **possibly ambiguous user question**

Your task is to **rewrite the user question to be fully self-contained and unambiguous**, so it can be interpreted without needing the background or prior conversation.

Instructions:
- Expand vague references (e.g., "it", "that", "those", "my condition") using relevant details from the **background** or **conversation**.
- Do **not** add any external information.
- Keep the meaning and intent of the original question unchanged.
- Return **only** the rewritten question (no explanation or preamble).
- If the original question is already clear and self-contained, return it as is.

---

# User Background:
{ptkb}

# Conversation History:
{history}

# User Question:
{User_Question}
"""

RE_PROMPT = """You are given some background of the user and conversation history between a user and a system.

Your task is to generate a **concise and informative one-sentence response** to the last user question based on the conversation history.

- Keep your answer specific and grounded in the context.
- Do not generate explanations or multiple options.
- If no history is available, return an empty string.

# User Background:
{ptkb}

# Conversation History:
{history}
"""

PR_PROMPT = """You are given some background of the user and conversation history between a user and a system, where the answer is not clear, and along with a new user question.

The final answer to the question is not available. Your goal is to generate a **plausible, one-sentence pseudo answer** that could logically follow from the context.

- Your answer should be concise (no more than 20 words).
- Do **not** include extra context or disclaimers.

# User Background:
{ptkb}

# Conversation History:
{history}

# New User Question:
{User_Question}
"""

TS_PROMPT = """You are given the conversation history between a user and a system, followed by a new user question.

Your task is to decide whether the new question **continues the same topic** or **starts a new topic**.

Respond strictly with one of the following two options:
- "old_topic" (if the question is contextually related to the previous turns)
- "new_topic" (if the question introduces a new topic)

If there is no conversational history provided, return "new_topic".

# Conversation History:
{history}

# New User Question:
{User_Question}
"""

HS_PROMPT = """You are given the conversation history between a user and a system.

Your task is to generate a paragraph that summarizes the entire dialogue.
- Write **one sentence** for each user-system pair. If any of these is missing, skip that part.
- Preserve the core facts and user intent.
- Avoid repetition or irrelevant details.

# Conversation History:
{enhanced_history}
"""

REWRITE_PROMPT = '''You are given:
- a set of user background,
- a conversation history between a user and a system,
- a clarified user question,
- and a pseudo response (not real, only for format reference).
- user preferences extracted from the background and history.

Your task is to rewrite the question into a **fully self-contained and unambiguous search query**, incorporating only the **necessary and helpful context** from the user background and conversation history.

Guidelines:
- Only include information that is **essential to interpret or disambiguate** the question.
- Consider include details (e.g., user location, preferences, conditions, etc...) **if they may be relevant** for disambiguating or contextualizing the question.
- If the question’s answer depends on the user’s identity (e.g., nationality, dietary restrictions, etc...), make sure to reflect that information.
- Do **not** invent or infer from the pseudo response. It is only provided to illustrate format, not content.
- Keep the rewritten query concise, natural, and focused.

If there is no conversation history, use only the **most relevant** user background details to clarify the question.

Output a JSON object in the format: {{"query": "..."}}  
No explanation or extra text.

# User Background:
{ptkb}

# Conversation History:
{enhanced_history}

# Clarified Question:
{disambiguated_question}

# Pseudo Response (Do not incorporate pseudo response facts into the query. It’s only provided as formatting aid.):
{pseudo_response}

# User Preferences:
{preference}

# Reminder:
When rewriting, use the user's background to reflect their interests or preferences — e.g., if the user enjoys hiking or art, suggest places matching those.
'''

RESPONSE_PROMPT = '''
Your task is to provide a direct, concise, and informative answer to the user's query.

INSTRUCTIONS:
- MUST avoid irrelevant content or generic filler language.
- COVER all key aspects of the user's question to ensure completeness.
- Be Factual and DIRECT: Provide a factual response. Do not ask for clarification. Do not say you don't know. Get straight to the point.

# User Background:
{ptkb}

# Conversation History:
{context}

# Self-Contained Query
{query}
'''