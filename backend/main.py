from dotenv import load_dotenv

load_dotenv()

import logging
import warnings
import os

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
warnings.filterwarnings(action="ignore", message=".*Failed to resolve*")
warnings.filterwarnings(action="ignore", message=".*Failed to establish*")
# Suppress detailed logs from urllib3 and Azure SDK
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure").setLevel(logging.WARNING)

# Optional: Suppress other noisy loggers
logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)

from typing import Any, Dict
import uuid

from backend.azure_ai_service import AzureAIService
from backend.azure_cosmos import AzureCosmos
from backend.prompts import Prompt


class RunAzureRagPipeline(AzureAIService, AzureCosmos, Prompt):
    def __init__(
        self,
        log_filename: str = "logs.log",
        log_format: str = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        log_datefmt: str = "%H:%M:%S",
        log_level: int = logging.DEBUG,
    ):
        logging.basicConfig(
            filename=log_filename,
            filemode="a",
            format=log_format,
            datefmt=log_datefmt,
            level=log_level,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger Initialised..")
        super().__init__(logger=self.logger)

    def get_prevoius_conversation(self, fetched_df, question_column_name="question"):
        previous_convo_string = ""
        for idx, row in fetched_df.iloc[::-1].reset_index(drop=True).iterrows():
            ques = row[question_column_name]
            if row["rephrased_question"]:
                ques = row["rephrased_question"]
            previous_convo_string += (
                f'User query {idx + 1}: {ques}\nAnswer: {row["answer"]}\n\n'
            )
        previous_convo_string = previous_convo_string.strip()
        return previous_convo_string

    async def query(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        file_name=None,
        file_names=None,
        project_code=None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Complete RAG query: search for relevant documents and generate answer

        Args:
            question: User's question
            user_id: Unique identifier for the user
            conversation_id: Unique identifier for the conversation
            session_id: Unique identifier for the session
            file_name: Name of the file being queried. If None, queries all files.
            file_names: List of file names to query. If provided, overrides file_name.
            project_code: Code of the project being queried. If None, queries all projects.
            top_k: Number of documents to retrieve for context

        Returns:
            Dictionary containing the answer and source documents
        """
        import json
        from datetime import datetime

        try:
            # Step 2: Initialise Cosmo DB data
            cosmos_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "project_code": project_code,
                "file_name": file_name,
                "question": question,
                "rephrased_question": "",
            }
            # print(f"Cosmos Data: {cosmos_data}")
            self.logger.info(f"Processing query: {question}")
            QUESTION_COL = "question"
            # Step 2: Fetch Past Questions from Cosmos DB
            fetched_df = self.read_cosmo_table(
                question_column=QUESTION_COL,
                user_id=user_id,
                session_id=session_id,
                file_name=file_name,
                project_code=project_code,
            )
            # print(fetched_df)
            if fetched_df is None or fetched_df is False:
                previous_convo_string = None
                self.logger.info(
                    "No previous questions found in Cosmos DB or error occurred"
                )
                # setting rephrased query to original query
                cosmos_data["rephrased_question"] = question
            else:
                previous_convo_string = self.get_prevoius_conversation(fetched_df)

                # #print("PREVIOUS CHAT:", previous_convo_string)
                rephrase_messages = self._query_rephrase_prompt(
                    query=question, previous_conversation=previous_convo_string
                )

                # #print(f"rephrase question:{rephrase_messages}")
                response = await self.get_openai_response(messages=rephrase_messages)
                rephrased_query = json.loads(response)["rephrased_query"]
                # #print(f"Rephrased_query: {rephrased_query}")

                if "not a follow-up question" not in rephrased_query.lower():
                    cosmos_data["rephrased_question"] = rephrased_query
                    QUESTION_COL = "rephrased_question"
                else:
                    cosmos_data["rephrased_question"] = question
                # print("&&" * 50)
                # #print(f"Previous Conversation: {previous_convo_string}")
                # print(f"Question Column: {QUESTION_COL}")
                # print(f"Rephrased query: {rephrased_query}")
                # print("&&" * 50)
            # Step 3: Check the intent of the question
            try:
                # print("Classifying intent of the question...")
                intent_messages = self.get_intent_prompt(
                    query=cosmos_data[QUESTION_COL]
                )
                intent = await self.get_openai_response(
                    messages=intent_messages, json_object=False
                )
                intent = intent.replace("`", "").strip()
                # print(f"Intent classified as: {intent}")
                intent = intent.strip().lower()
                self.logger.info(f"Intent classified as: {intent}")
            except Exception as e:
                self.logger.error(f"Intent classification failed: {str(e)}")
                intent = "other"  # Default to 'other' if intent classification fails

            if intent == "file_reference":
                # If the intent is file_reference, return available files
                available_files = await self.get_available_files()
                # print(f"Available files: {available_files}")
                if not available_files:
                    answer = "No files available for reference."
                else:
                    total_files = len(available_files)
                    filenames = [os.path.basename(f["value"]) for f in available_files]
                    file_list = "\n- " + "\n- ".join(filenames)
                    answer = f"I have access total {total_files} files:\n{file_list}\n\nYou can ask me about these files."

                self.logger.info("Intent is file_reference, returning available files")
                return {
                    "question": question,
                    "rephrased_question": cosmos_data[QUESTION_COL],
                    "answer": answer,
                    "references": "",
                    "source_documents": [],
                    "is_relevant": True,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }

            if intent == "english_grammar":
                # If the intent is english_grammar, return a default message
                self.logger.info("Intent is english_grammar, returning default message")
                return {
                    "question": question,
                    "rephrased_question": cosmos_data[QUESTION_COL],
                    "answer": "Sorry, I cannot help with word meanings or literature-related questions. Please ask something related to the uploaded documents.",
                    "source_documents": [],
                    "is_relevant": False,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            # print("Intent classification passed, proceeding with RAG pipeline")

            # Step 4: Extract filename from user query
            # #print("Extracting filename from user query...")
            # predicted_filename_prompt = self.extract_filename_from_user_query(
            #     query=cosmos_data[QUESTION_COL],
            #     available_refrences=await self.get_available_files())
            # predicted_filename = await self.get_openai_response(messages=predicted_filename_prompt,json_object=False)

            # Step 4: Search for relevant documents
            # Use file_names if provided, otherwise fall back to file_name
            search_filter = file_names if file_names is not None else file_name
            relevant_docs = await self.search_similar_documents(
                cosmos_data[QUESTION_COL], top_k, search_filter
            )

            # relevant_docs = []
            # for file_name in search_filter:
            #     sub_relevant_docs = await self.search_similar_documents(
            #         cosmos_data[QUESTION_COL], top_k, file_name
            #     )
            #     relevant_docs = relevant_docs + sub_relevant_docs

            # print(f"Relevant documents found: {len(relevant_docs)}")
            # for doc in relevant_docs:
            #     #print("RAG chunk content:", doc["content"][:500])

            # #print("*" * 50)
            # #print(len(relevant_docs))
            # #print(relevant_docs)
            # #print("*" * 50)

            # Step 5: Check if the query is relevant to the documents
            is_relevant = self._is_query_relevant(
                cosmos_data[QUESTION_COL], relevant_docs
            )
            # print(f"Is query relevant: {is_relevant}")
            if is_relevant is None:
                # print("[WARN] _is_query_relevant returned None. Forcing True.")
                is_relevant = True
            # Step 6: Generate answer using retrieved documents
            chat_model_prompt = self.get_chat_model_prompt(
                query=cosmos_data[QUESTION_COL],
                previous_convo_string=previous_convo_string,
                context_docs=relevant_docs,
            )
            # #print(f"Chat model prompt: {chat_model_prompt}")
            references = ""
            if not is_relevant:
                answer = """Thanks for your question! 😊 Unfortunately, I couldn’t find any relevant information in the uploaded documents to answer your question accurately. If you have any specific document that mentions this topic, feel free to upload it and I’ll gladly help!"""
            else:
                try:
                    answer = await self.get_openai_response(
                        messages=chat_model_prompt, json_object=True
                    )
                    answer = answer.replace("```", "")
                    # print(f"Generated answer: {answer}")
                    parsed = json.loads(answer)  # -->
                    answer = parsed.get("Answer", "").strip()  # -->
                    references = parsed.get("References", "").strip()  # -->
                    self.logger.info("Generated answer using RAG pipeline")
                    # print(f"Answer: {answer}")
                    # print(f"References: {references}")
                except Exception as e:
                    self.logger.error(f"Error generating answer: {str(e)}")
                    raise

            # Step 5: Return complete response with relevance flag
            response = {
                "question": question,
                "rephrased_question": cosmos_data[QUESTION_COL],
                "answer": answer,
                "references": references,
                "source_documents": relevant_docs if is_relevant else [],
                "is_relevant": is_relevant,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            # cosmos_data = {**cosmos_data, **response.items()}
            for key, value in response.items():
                cosmos_data[key] = value
            self.logger.info("RAG query completed successfully")
            # #print(response)
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    async def run(
        self,
        create_new_index: bool = False,
        upload_to_blob: bool = False,
        index_document: bool = False,
        pdf_path: str = None,
        blob_name: str = None,
        blob_kwargs: dict = None,
    ):
        # print("Azure Rag Pipeline")
        if create_new_index:
            # Step 1: Create the search index
            # print("Creating search index...")
            await self.create_search_index()

        if upload_to_blob:
            if pdf_path is None:
                raise Exception("pdf_path cannot be None")
            if blob_name is None:
                raise Exception("blob_name cannot be None")
            # Step 2: Upload a PDF to blob storage (replace with your PDF path)
            # print("Uploading PDF to blob storage...")
            # pdf_path = "502-UCV-Agreement_with_Hexacom.pdf"
            # blob_name = f"Categories/502/{pdf_path}"
            if blob_kwargs is None:
                blob_kwargs = {}

            blob_name, blob_url = self.upload_pdf_to_blob(
                file_path=pdf_path, blob_name=blob_name, **blob_kwargs
            )
        if index_document:
            if upload_to_blob:
                # Step 3: Index the document
                # print("Indexing document...")
                await self.index_document(blob_name)
            else:
                print("Uploading is Required to index document...")

    # async def run(self, create_new_index: bool=False,
    #               upload_to_blob: bool=False, index_document: bool=False,
    #               pdf_path: str=None, blob_name: str=None, blob_kwargs: dict=None):
    #     print("Azure Rag Pipeline")
    #     if create_new_index:
    #         # Step 1: Create the search index
    #         print("Creating search index...")
    #         await self.create_search_index()

    #     if upload_to_blob:
    #         if pdf_path is None:
    #             raise Exception("pdf_path cannot be None")
    #         if blob_name is None:
    #             raise Exception("blob_name cannot be None")
    #         # Step 2: Upload a PDF to blob storage (replace with your PDF path)
    #         print("Uploading PDF to blob storage...")
    #         # pdf_path = "502-UCV-Agreement_with_Hexacom.pdf"
    #         # blob_name = f"Categories/502/{pdf_path}"
    #         if blob_kwargs is None:
    #             blob_kwargs = {}
    #         blob_path = self.upload_pdf_to_blob(file_path=pdf_path, blob_name=blob_name, **blob_kwargs)
    #     if index_document:
    #         if upload_to_blob:
    #             # Step 3: Index the document
    #             print("Indexing document...")
    #             await self.index_document(blob_path)
    #         else:
    #             print("Uploading is Required to index document...")


if __name__ == "__main__":
    import asyncio

    obj = RunAzureRagPipeline()
    asyncio.run(obj.run(create_new_index=True))
    # asyncio.run(
    #     obj.query(
    #         question="Why do you think the USOF might have extended the bid submission and technical bid opening dates?",
    #         user_id="f619b0f6ed6605adc1c624a3f6091bcb",
    #         conversation_id="3844ee18-39f5-459e-8983-1ad39cdd7edc",
    #         session_id="4e773c72-570f-4615-8307-e5e0113df624"
    #     )
    # )
