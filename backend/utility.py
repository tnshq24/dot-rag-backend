import os
import re
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any
import tiktoken
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Utility:
    def __init__(self, logger):
        load_dotenv(override=True)
        self.logger = logger

    def _get_env_variables(self, env_variable: str, default_value=None):
        env_value = os.getenv(env_variable, default_value)
        if env_value is None:
            self.logger.error("Environment variable '{}' is not set".format(env_variable))
        else:
            self.logger.info("Environment variable '{}' is set".format(env_variable))
        return os.getenv(env_variable)

    # def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    #     """
    #     Split text into overlapping chunks for better retrieval

    #     Args:
    #         text: Input text to chunk
    #         chunk_size: Maximum size of each chunk
    #         overlap: Number of characters to overlap between chunks

    #     Returns:
    #         List of text chunks
    #     """
    #     # Clean the text first
    #     text = text.strip()
    #     if not text:
    #         return []

    #     chunks = []
    #     start = 0

    #     while start < len(text):
    #         # Calculate the end position for this chunk
    #         end = start + chunk_size

    #         # If this isn't the last chunk, try to break at a word boundary
    #         if end < len(text):
    #             # Find the last space before the end position
    #             while end > start and text[end] != " ":
    #                 end -= 1

    #             # If no space found, use the original end position
    #             if end == start:
    #                 end = start + chunk_size

    #         # Extract the chunk
    #         chunk = text[start:end].strip()

    #         # Only add non-empty chunks with reasonable length
    #         if chunk and len(chunk) > 10:  # Skip very short chunks
    #             chunks.append(chunk)

    #         # Move start position for next chunk (with overlap)
    #         start = end - overlap

    #         # Ensure we don't go backwards and avoid infinite loops
    #         if start <= 0 or start >= len(text):
    #             break

    #     return chunks
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval using langchain RecursiveCharacterTextSplitter

        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        def count_tokens(text: str) -> int:
            """
            Count the number of tokens in a text string.
            """
            # Assuming 1 token = 4 characters on average
            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens = encoding.encode(text)
            return len(tokens) 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=count_tokens,
            add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        return chunks
    # def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    #     try:
    #         text = text.strip()
    #         if not text:
    #             return []

    #         if chunk_size <= 0:
    #             raise ValueError("chunk_size must be > 0")
    #         if overlap < 0:
    #             raise ValueError("overlap must be >= 0")
    #         if overlap >= chunk_size:
    #             raise ValueError("overlap must be smaller than chunk_size")

    #         chunks: List[str] = []
    #         n = len(text)
    #         start = 0

    #         while start < n:
    #             end = min(n, start + chunk_size)

    #             # Try to cut at a word boundary if we're not at the end
    #             if end < n:
    #                 split_pos = text.rfind(" ", start + 1, end)
    #                 if split_pos > start:  # found a space after start
    #                     end = split_pos

    #             chunk = text[start:end].strip()
    #             if chunk:
    #                 chunks.append(chunk)

    #             if end == n:            # consumed all text
    #                 break

    #             # Compute next start with overlap and guarantee forward progress
    #             next_start = end - overlap
    #             if next_start <= start:  # safety guard against infinite loops
    #                 next_start = end

    #             start = next_start

    #         return chunks
    #     except Exception as e:
    #         print(f"Error chunking text: {str(e)}")
    #         # return []


    def _is_query_relevant(
            self, question: str, relevant_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if the query is relevant to the uploaded documents

        Args:
            question: User's question
            relevant_docs: Retrieved documents

        Returns:
            True if query is relevant, False otherwise
        """
        import re
        # Define irrelevant query patterns
        irrelevant_patterns = [
            r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
            r"\b(how are you|how do you do)\b",
            r"\b(what is your name|who are you)\b",
            r"\b(thank you|thanks)\b",
            r"\b(bye|goodbye|see you)\b",
            r"\b(what time|what day|what date)\b",
            r"\b(weather|temperature)\b",
            r"\b(joke|funny|humor)\b",
            # r"\b(help|support)\b",
            r"\b(menu|options)\b",
        ]

        # Check if query matches irrelevant patterns
        question_lower = question.lower().strip()
        for pattern in irrelevant_patterns:
            if re.search(pattern, question_lower):
                return False

        # Check if we have relevant documents with good scores
        if not relevant_docs:
            return False

        # # Check if any document has a good relevance score
        # max_score = max(doc.get("score", 0) for doc in relevant_docs)
        # return max_score > 0.3  # Adjust threshold as needed

    def sanitize_document_key(self, key: str) -> str:
        # Replace any sequence of non-allowed chars with a single underscore
        sanitized = re.sub(r'[^A-Za-z0-9_=-]', '_', key)
        return sanitized




