import traceback
import os
import re
from azure.core.credentials import AzureKeyCredential
from backend.azure_blob_storage import AzureBlobStorage
from backend.form_rec import extract_text_and_tables, convert_to_txt
import tiktoken
class AzureDocumentIntelligence(AzureBlobStorage):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()

    def __initialize_services(self):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = self._get_env_variables(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
        )
        AZURE_DOCUMENT_INTELLIGENCE_API_KEY = self._get_env_variables(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
        )
        if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
            raise Exception("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable not set")
        if not AZURE_DOCUMENT_INTELLIGENCE_API_KEY:
            raise Exception("AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable not set")

        try:
            self.AZURE_DOCUMENT_INTELLIGENCE_CLIENT = DocumentIntelligenceClient(
                endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_API_KEY),
            )
            self.logger.info("AZURE_DOCUMENT_INTELLIGENCE_CLIENT initalized successfully")
        except Exception as e:
            self.logger.error("AZURE_DOCUMENT_INTELLIGENCE_CLIENT initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

    # def _extract_using_document_intelligence(self, blob_name: str, return_raw: bool = False):
    #     """
    #     Extract text content from a PDF stored in Azure Blob Storage

    #     Args:
    #         blob_name: Name of the PDF blob in storage

    #     Returns:
    #         List of dictionaries containing page text and metadata
    #     """
    #     from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    #     try:
    #         # Get the blob client for the PDF file
    #         blob_client = self.get_azure_blob_client(blob_name=blob_name)

    #         pdf_reader = self.AZURE_DOCUMENT_INTELLIGENCE_CLIENT.begin_analyze_document(
    #             "prebuilt-layout", AnalyzeDocumentRequest(url_source=blob_client.url)
    #         )
    #         pdf_reader = pdf_reader.result()
    #         if return_raw:
    #             return pdf_reader
            
    #         pages_content = []
    #         for page in pdf_reader.pages:
    #             # Combine all text from the page
    #             page_text = ""

    #             # Extract text from lines (preserves reading order)
    #             if hasattr(page, "lines") and page.lines:
    #                 for line in page.lines:
    #                     page_text += line.content + "\n"

    #             # If no lines, try paragraphs
    #             elif hasattr(pdf_reader, "paragraphs"):
    #                 page_paragraphs = [
    #                     p
    #                     for p in pdf_reader.paragraphs
    #                     if hasattr(p, "bounding_regions")
    #                        and any(
    #                         br.page_number == page.page_number
    #                         for br in p.bounding_regions
    #                     )
    #                 ]
    #                 for paragraph in page_paragraphs:
    #                     page_text += paragraph.content + "\n\n"

    #             # Only include pages with meaningful content
    #             if page_text.strip():
    #                 pages_content.append(
    #                     {
    #                         "page_number": page.page_number,
    #                         "content": page_text.strip(),
    #                         "filename": blob_name,
    #                     }
    #                 )
    #         dir_name = "/Users/sakhiagarwal/Downloads/dot_rag_pipeline 2/extracted_text-with-filter"
    #         filename = blob_name.split("/")[-1].replace(".pdf", "")
    #         file_path = os.path.join(dir_name, f"{filename}.txt")
    #         os.makedirs(dir_name, exist_ok=True)
    #         with open(file_path, "w", encoding="utf-8") as f:
    #             for page in pages_content:
    #                 f.write(f"Page {page['page_number']}:\n{page['content']}\n\n")
    #         print(f"Extracted text saved to {file_path}")
    #         return pages_content
    #     except Exception as e:
    #         self.logger.error(f"Error extracting text from PDF: {str(e)}")
    #         raise

    def split_text_files_in_folder(self, filename: str, text:str) -> list:
        result = []
        pages = re.split(r'(=== Page \d+ ===)', text)
        # Combine marker and content for each page
        for i in range(1, len(pages), 2):
            page_marker = pages[i]
            page_content = pages[i+1] if i+1 < len(pages) else ''
            if self.count_tokens(page_content.strip()) <= 20:
                continue 
            # Extract page number from marker
            page_no_match = re.search(r'=== Page (\d+) ===', page_marker)
            page_no = str(page_no_match.group(1)) if page_no_match else None
            sanitized = re.sub(r'[^A-Za-z0-9_=-]', '_', filename)
            chunk_id = f"{sanitized}_p{page_no}"
            result.append({
                'content': page_content.strip(),
                'page_number': str(page_no),
                'filename': filename,
                'chunk_id': chunk_id
            })
        # save all chunks
        folder_path = filename.replace('.pdf', '_chunks')
        os.makedirs(folder_path, exist_ok=True)
        for chunk in result:
            chunk_file_path = os.path.join(folder_path, f"{chunk['chunk_id']}.txt")
            with open(chunk_file_path, "w", encoding="utf-8") as f:
                f.write(chunk['content'])
        return result
    
    def count_tokens(self, text: str) -> int:
            """
            Count the number of tokens in a text string.
            """
            # Assuming 1 token = 4 characters on average
            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens = encoding.encode(text)
            return len(tokens)

    def _extract_using_document_intelligence(self, blob_name: str, return_raw: bool = False):
        """
        Extract text content from a PDF stored in Azure Blob Storage

        Args:
            blob_name: Name of the PDF blob in storage

        Returns:
            List of dictionaries containing page text and metadata
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        try:
            #print(f"Extracting text from PDF: {blob_name}")
            # Get the blob client for the PDF file
            blob_client = self.get_azure_blob_client(blob_name=blob_name)

            pdf_reader = self.AZURE_DOCUMENT_INTELLIGENCE_CLIENT.begin_analyze_document(
                "prebuilt-layout", AnalyzeDocumentRequest(url_source=blob_client.url)
            )
            pdf_reader = pdf_reader.result()
            if return_raw:
                return pdf_reader
            
            content = extract_text_and_tables(pdf_reader.as_dict(), output="markdown")
            text_content = convert_to_txt(content)
            chunks = self.split_text_files_in_folder(blob_name, text_content)
            return chunks
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise