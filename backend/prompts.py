from backend.utility import Utility

class Prompt(Utility):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)

    def _query_rephrase_prompt(self, query: str, previous_conversation: str):
        """
        Generates a prompt for rephrasing a query.
        Args:
            query (str): The query to be rephrased.
            previous_conversation (str): The previous conversation.

        Returns:
            List[Dict[str, str]]: A list of messages containing the prompt and the user query.
        """
        prompt = """You are a query rephrasing tool that rephrases follow-up questions into standalone questions (without any modification in entity values) which can be understood independently without relying on previous question and answer.

Objective: Analyze the chat history enclosed within triple backticks, carefully to create standalone question independent of terms like 'it', 'that', etc.
For queries that are not a follow-up ones or not related to the conversation, you will respond with a predetermined message: 'Not a follow-up question'

**Critical Instructions:**
1. Analyze the chat history enclosed within triple backticks, Classify the Current query as DEPENDENT (needs HISTORY) or INDEPENDENT.
2. If DEPENDENT, produce a fully self-contained question by inserting only the missing subject/time/entities from HISTORY.
3. If INDEPENDENT , return the 'Not a follow-up question'.
    - If the CURRENT INDEPENDENT query uses abbreviations, acronyms, nicknames, file short-codes, or numeric shorthands that are explicitly expanded in the Previous conversation. Alter the CURRENT query to use the full form or full name as per the Previous conversation.

## CHAT HISTORY:
```
{previous_conversation}
```

## Output Format:
    A JSON dict with 1 key:
        - 'rephrased_query'(str): It Contains the rephrased query formed by following the above instructions."""
        prompt = prompt.format(previous_conversation=previous_conversation)

        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append(
        {
            "role": "user",
            "content": f"""Query: {query}\nGiven the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.""",
        }
    )
        return messages

    def get_intent_prompt(self, query):
        check_prompt = f"""Question: "{query}"\nClassify the intent of the query into one of these categories:
1. english_grammar → Questions about English language, grammar, literature, or meanings of words.

2. file_reference → Only when the question is about the **count or existence** of uploaded reference documents or files. Example phrases include:
- "How many files are there?"
- "How many documents have been uploaded?"
- "How many reference tables do you have?"

**Do NOT classify as `file_reference` if the question is about the **content, information inside, purpose, comparison, or usage** of documents. Those belong to `other`.**

3. other → Any other type of question, including those about:
- Content or details of documents (e.g., "What is in the tender file?")
- Comparing or summarizing documents
- Legal, technical, financial, or project-related questions
- Anything that is not purely about the number of uploaded files or reference documents

Answer only with one of: `english_grammar`, `file_reference`, `other`."""

        messages = [
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": check_prompt},
        ]
        return messages

    def get_chat_model_prompt(self, query: str ,previous_convo_string: str, context_docs: list):
        import os

        context = "\n\n".join(
            [
                f"Document (filename): `{os.path.basename(doc['filename']).strip()}` (Page {doc['page_number']})\n{doc['content'].strip()}\n---"
                for doc in context_docs
            ]
        )
        # Step 2: Create the prompt for the chat model
        system_prompt = """You are an expert assistant with access to a set of uploaded legal Tender documents.
Follow these rules strictly while responding:
1. Context and Relevance: Answers must ONLY be based on the provided context (documents) and previous chat history. - Do NOT use outside knowledge or assumptions.
Do NOT reference all documents retrieved. If only one or two are needed, use only those.

2. Document References: Always cite which document and page number supports your statements. Clearly distinguish between documents that support **factual claims** vs **inferred interpretations**.

3. Inference vs Factual Claims: If the answer is found **explicitly** in the documents, clearly present it as a fact.
    - If the answer is **inferred** (not directly stated), clearly label it using phrases like:
    - "Based on inference..."
    - "While not explicitly stated, this can be interpreted as..."
    - "It is reasonable to infer from..."
❗ NEVER present inferred information as if it is directly written in the document.
`
4. Language Guidelines:
- Use cautious language when inferring. Avoid overconfident terms like “will result in” unless the document clearly states it.
- Prefer terms like “may lead to,” “could result in,” or “might imply” for inferred content.

5. Multiple Document References: Provide document references only if the answer is explicitly found within the documents, include references at the end of the answer Also , below format given between triple backticks. If clarification is needed, do NOT mention document references prematurely. Ask for clarification clearly without citing any documents.

4. Out-of-Scope Queries: If the user's query does not pertain to any content within the uploaded documents, explicitly state that the query is outside the scope of available documents. ***Do NOT mention any document references in such cases.*** Use this response template when information is unavailable:
"Thank you for your question! However, after reviewing the provided documents, I couldn’t find relevant information to accurately answer your query. If this topic is covered in any other document, please upload it, and I'll be happy to assist further.

###Output Format:
- Your response must be a valid JSON format with two string keys.
    - "Answer": "Your complete response as a properly formatted string with bullet points, line breaks, and formatting preserved as plain text",
    - "References": "filename.pdf, Page: X"

Past Conversation (If any):
{previous_conversation}"""

        user_prompt = f"""Context Documents:
{context.strip()}

Question: {query}

Please provide a detailed answer based on the context documents above."""
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    previous_conversation=previous_convo_string
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        #print("^" * 20)
        #print(user_prompt)
        #print("^" * 20)
        return messages


    def extract_filename_from_user_query(self, query: str,available_refrences:str) -> str:
        """
        Extracts the filename from a user query if it contains a file reference.
        
        Args:
            query (str): The user query containing a file reference.
        
        Returns:
            str: The extracted filename or an empty string if no file reference is found.
        """
        system_prompt ="""You are an intelligent assistant that classifies which document/file is most relevant for a given user query.

Here is a list of available reference documents/files:
{available_refrences}

You will be given a user query. Your job is to identify only one filename from the above list that is most relevant to answer the query. Check the acronyms and abbreviations in the query.
Only return the exact filename path from the list. Do not generate any other explanation or text.
If none of the documents is relevant, output exactly None

Below are a few examples to guide your behavior:

### Examples:

**User Query:** What is the agreement with RJIO for Andhra Pradesh and Odisha?  
**Answer:** Categories/7287/7287-Aspirational-Agreement_with_RJIO_Andhra_Pradesh_Chhattisgarh_&_Odisha.pdf

**User Query:** What are the rules under Digital Bharat Nidhi?  
**Answer:** Categories/Acts/Administration of DBN Rules 2024.pdf
 """

        user_prompt = f"""User Query: {query}\nPlease extract the filename from the available references that is most relevant to this query."""
#         system_prompt = """[ROLE]
# You are **DocumentSelector**, an AI assistant whose sole task is to choose the single most relevant reference document for each user query.

# # [REFERENCE FILES]
# # {available_refrences}

# # [TASK]
# # 1. Read the *User Query*.
# # 2. Select **one** filename from the list that best answers the query. Check the acronyms and abbreviations in the query.
# # 3. Output *only* the exact filename (including full path and extension).
# # 4. If none of the documents is relevant, output exactly None.  
# # 5. Absolutely no extra text, numbering, or punctuation.

# # FYI (Acronyms):
# # - UCV means Uncovered Villages
# # - DBN means Digital Bharat Nidhi

# # [OUTPUT FORMAT]
# # Output **only** the chosen filename"""
        
#         user_prompt = f"""User Query: {query}"""
        messages = [
            {"role": "system", "content": system_prompt.format(available_refrences=available_refrences)},
            {"role": "user", "content": user_prompt},
        ]       
        #print("^" * 20)
        #print(system_prompt.format(available_refrences=available_refrences))
        #print(user_prompt)
        #print("^" * 20)
        return messages










