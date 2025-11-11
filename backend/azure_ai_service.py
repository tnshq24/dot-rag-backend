import os
import traceback
from typing import List, Dict, Any, Optional, Union

from azure.core.credentials import AzureKeyCredential


from backend.azure_open_ai import AzureOpenAI
from backend.azure_document_intelligence import AzureDocumentIntelligence

class AzureAIService(AzureOpenAI, AzureDocumentIntelligence):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()

    def __initialize_services(self):
        """
        Initialise Azure Services client
        Return : AZURE_SERVICE_INDEX_CLIENT, AZURE_SEARCH_CLIENT
        """
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents import SearchClient

        AZURE_SEARCH_SERVICE_NAME = self._get_env_variables("AZURE_SEARCH_SERVICE_NAME")
        AZURE_SEARCH_ADMIN_KEY = self._get_env_variables("AZURE_SEARCH_ADMIN_KEY")
        self.AZURE_SEARCH_INDEX_NAME = self._get_env_variables("AZURE_SEARCH_INDEX_NAME")

        if AZURE_SEARCH_SERVICE_NAME:
            AZURE_SEARCH_ENDPOINT = f"https://{AZURE_SEARCH_SERVICE_NAME}.search.windows.net"
        else:
            raise Exception("AZURE_SEARCH_SERVICE_NAME environment variable not set")
        if not AZURE_SEARCH_ADMIN_KEY:
            raise Exception("AZURE_SEARCH_ADMIN_KEY environment variable not set")
        if not self.AZURE_SEARCH_INDEX_NAME:
            raise Exception("AZURE_SEARCH_INDEX_NAME environment variable not set")
        try:
            self.AZURE_SERVICE_INDEX_CLIENT = SearchIndexClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
            )
            self.logger.info("AZURE_SERVICE_INDEX_CLIENT initalized successfully")
        except Exception as e:
            self.logger.error("AZURE_SERVICE_INDEX_CLIENT initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

        try:
            # SearchClient performs search operations on the index
            self.AZURE_SEARCH_CLIENT = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=self.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
            )
            self.logger.info("AZURE_SEARCH_CLIENT initalized successfully")
        except Exception as e:
            self.logger.error("AZURE_SERVICE_CLIENT initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

    async def create_search_index(self):
        """
        Create the Azure AI Search index with vector search capabilities
        This index will store document chunks and their vector embeddings
        """
        from azure.search.documents.indexes.models import (
            SearchIndex,
            SearchField,
            SearchFieldDataType,
            VectorSearch,
            VectorSearchProfile,
            HnswAlgorithmConfiguration,
            VectorSearchAlgorithmKind,
            SearchableField,
            SimpleField,
        )

        # Define the search index fields
        # These fields define the structure of documents in our search index
        fields = [
            # Unique identifier for each document chunk
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            # Original filename of the PDF
            SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True,facetable=True),
            # Text content of the document chunk
            SearchableField(name="content", type=SearchFieldDataType.String),
            # Page number where this chunk appears
            SearchableField(name="page_number", type=SearchFieldDataType.String, searchable=True),
            # Timestamp when the document was processed
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset),
            # Vector embedding of the content (1536 dimensions for Ada model)
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada embedding model dimension
                vector_search_profile_name="my-vector-profile",
            ),
        ]

        # Configure vector search settings
        # This enables similarity search using vector embeddings
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-config",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={
                        "m": 4,  # Number of bi-directional links for every new element
                        "efConstruction": 400,  # Size of the dynamic candidate list
                        "efSearch": 500,  # Size of the dynamic candidate list used during search
                        "metric": "cosine",
                    },
                )
            ],
        )

        # Create the search index
        index = SearchIndex(
            name=self.AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search
        )

        try:
            # Create or update the index in Azure AI Search
            result = self.AZURE_SERVICE_INDEX_CLIENT.create_or_update_index(index)
            self.logger.info(f"Search index '{self.AZURE_SEARCH_INDEX_NAME}' created successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error creating search index: {str(e)}")
            raise

    def __read_all_page_content(self, pdf_content, blob_name):
        import PyPDF2
        from io import BytesIO

        pdf_stream = BytesIO(pdf_content)
        # #print("Streaming")
        # Extract text from each page using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        pages_content = []
        for page_num, page in enumerate(pdf_reader.pages):
            # Extract text from the current page
            text = page.extract_text()
            # Only include pages with meaningful content
            if text.strip():
                pages_content.append(
                    {
                        "page_number": page_num + 1,
                        "content": text.strip(),
                        "filename": blob_name,
                    }
                )
        return pages_content


    def _extract_text_from_pdf_blob(self, blob_name: str) -> List[Dict[str, Any]]:
        """
        Extract text content from a PDF stored in Azure Blob Storage

        Args:
            blob_name: Name of the PDF blob in storage

        Returns:
            List of dictionaries containing page text and metadata
        """

        try:
            #print("HEY")
            pdf_content = self.get_pdf_content_from_blob(blob_name=blob_name)
            #print("HEY1")
            pages_content = self.__read_all_page_content(pdf_content=pdf_content, blob_name=blob_name)
            self.logger.info(f"Extracted text from {len(pages_content)} pages")
            return pages_content
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise


    def __extract_chunks_from_page_content(self, pages_content, blob_name):
        all_chunks = []
        for page_data in pages_content:
            # Skip pages with very little content
            # if len(page_data["content"].strip()) < 50:
            #     continue

            # Split page content into smaller chunks
            chunks = self.chunk_text(page_data["content"])

            # Create chunk documents with metadata
            for i, chunk in enumerate(chunks):
                # Clean and validate chunk content
                chunk = chunk.strip()
                # if len(chunk) < 20:  # Skip very short chunks
                #     #print(f"Skipping short chunk on page {page_data['page_number']}: {chunk}")
                #     continue

                # Create a safe, unique ID
                # safe_filename = blob_name.replace(".", "_").replace(" ", "_").replace("/", "=")
                safe_filename = self.sanitize_document_key(blob_name)
                chunk_id = f"{safe_filename}_p{page_data['page_number']}_c{i}"

                chunk_doc = {
                    "content": chunk,
                    "filename": blob_name,
                    "page_number": page_data["page_number"],
                    "chunk_id": chunk_id,
                }
                all_chunks.append(chunk_doc)
        return all_chunks


    async def __upload_documents_to_index(self, all_chunks, blob_name, batch_size: int=10):
        def get_document_batch(batch_chunks, embeddings):
            documents = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                doc = {
                    "id": chunk["chunk_id"],
                    "filename": chunk["filename"],
                    "content": chunk["content"],
                    "page_number": chunk["page_number"],
                    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "content_vector": embedding,
                }
                documents.append(doc)
            return documents

        from datetime import datetime
        total_indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i: i + batch_size]

            # Step 4: Generate embeddings for this batch
            # chunk_texts = [chunk["content"] for chunk in batch_chunks]
            chunk_text_with_metadata = [f"Filename: {os.path.basename(chunk['filename'])}\nPage number: {chunk['page_number']}\n{chunk['content']}" for chunk in batch_chunks]
            ##print(f"<CHUNK>")
            ##print(chunk_text_with_metadata)
            ##print(f"</CHUNK>")
            embeddings = await self.generate_embeddings(chunk_text_with_metadata)

            # Step 5: Prepare documents for indexing
            documents = get_document_batch(batch_chunks=batch_chunks, embeddings=embeddings)
            # Step 6: Upload this batch to Azure AI Search
            try:
                result = self.AZURE_SEARCH_CLIENT.upload_documents(documents)
                successful_uploads = sum(1 for r in result if r.succeeded)
                total_indexed += successful_uploads

                # Log any failures
                for r in result:
                    if not r.succeeded:
                        self.logger.warning(
                            f"Failed to index document {r.key}: {r.error_message}"
                        )
            except Exception as e:
                self.logger.error(f"Error indexing batch: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        self.logger.info(f"Successfully indexed {total_indexed} chunks from {blob_name}")
        return total_indexed


    async def index_document(self, blob_name: str):
        """
        Process a PDF document and index it in Azure AI Search

        Args:
            blob_name: Name of the PDF blob to process
        """
        try:
            # Step 1: Extract text from PDF
            self.logger.info(f"Processing document: {blob_name}")
            # pages_content = self._extract_text_from_pdf_blob(blob_name)
            # if len(pages_content) == 0:
            #     #print(f"No content extracted from {blob_name} using pyPDF2, now trying Azure Document Intelligence")
            #     pages_content = self._extract_using_document_intelligence(blob_name=blob_name)

            # if not pages_content:
            #     self.logger.warning(f"No content extracted from {blob_name}")
            #     #print("="*100)
            #     #print(f"No content extracted from {blob_name}")
            #     #print("="*100)
            #     return

            # # Step 2: Chunk the text content
            # # #print(f"Page content: {pages_content}")
            # #print(f"Total pages extracted: {len(pages_content)}")
            # all_chunks = self.__extract_chunks_from_page_content(pages_content=pages_content, blob_name=blob_name)
            # #print(f"Total chunks created: {len(all_chunks)}")
            # if not all_chunks:
            #     self.logger.warning(f"No valid chunks created from {blob_name}")
            #     #print("=" * 100)
            #     #print(f"No valid chunks created from {blob_name}")
            #     #print("=" * 100)
            #     return
            all_chunks = self._extract_using_document_intelligence(blob_name=blob_name)
            self.logger.info(f"Created {len(all_chunks)} chunks from {blob_name}")
            # Step 3: Process chunks in smaller batches to avoid overwhelming the API
            # Process 10 chunks at a time
            total_indexed = await self.__upload_documents_to_index(
                all_chunks=all_chunks, blob_name=blob_name, batch_size=10
            )
            return {"indexed_chunks": total_indexed, "total_chunks": len(all_chunks)}
        except Exception as e:
            self.logger.error(f"Error indexing document: {str(e)}")
            raise
        
    async def get_available_files(self):
        try:
            results = self.AZURE_SEARCH_CLIENT.search(
                search_text="*",
                facets=["filename,count:1000"],
                top=0
            )
            results = results.get_facets().get("filename", [])
            # filenames = "\n".join(
            #     [f"{idx+1}. {item['value']}" for idx, item in enumerate(results)]
            # )
            return results
        except Exception as e:
            self.logger.error(f"Error fetching indexed file summary: {str(e)}")
            return []


    async def search_similar_documents(
        self, query: str, top_k: int = 8, filename_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query using vector similarity

        Args:
            query: Search query text
            top_k: Number of similar documents to return
            filename_filter: Single filename string or list of filenames to filter by

        Returns:
            List of similar documents with scores and blob URLs
        """
        from azure.search.documents.models import VectorizedQuery
        from azure.search.documents.models import VectorFilterMode
        try:
            # Step 1: Generate embedding for the search query
            # #print(f"query : {query}")
            query_embedding = await self.generate_embeddings([query])
            query_vector = query_embedding[0]

            # Step 2: Create a vectorized query for Azure AI Search
            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
            )
            # filter_expression = None
            # if filename_filter is None or filename_filter.lower() == "none":
            #     #print("No filename filter applied")
            
            # if filename_filter:
            #     #print(f"filename: {filename_filter}")
            #     #print(repr(filename_filter))
            #     filter_expression = f"filename eq '{filename_filter}'"

            # #print(f"Filtering results by filename: {filename_filter}") 
            # filter_expression = None

            # # Normalize the input to handle both None and the string "None"
            # if filename_filter is None or filename_filter.lower() == "none":
            #     #print("No filename filter applied")
            # else:
            #     #print(f"filename: {filename_filter}")
            #     #print(repr(filename_filter))
            #     filter_expression = f"filename eq '{filename_filter}'"
            #print(f"filename_filter: {filename_filter}")

            if filename_filter is None:
                #print("No filename filter applied")
                filter_expression = None
            elif isinstance(filename_filter, list):
                # Handle multiple filenames
                if len(filename_filter) == 0:
                    filter_expression = None
                elif len(filename_filter) == 1:
                    filter_expression = f"filename eq '{filename_filter[0]}'"
                else:
                    # Create OR filter for multiple filenames
                    filename_conditions = [f"filename eq '{fname}'" for fname in filename_filter]
                    # filter_expression = " or ".join(filename_conditions)
                    filter_expression = " or ".join(filename_conditions)
            else:
                # Handle single filename string
                filter_expression = f"filename eq '{filename_filter}'"
            #print(f"Filtering results by filename: {filter_expression}")


            search_results = self.AZURE_SEARCH_CLIENT.search(
                search_text=query,  # We're doing pure vector search
                vector_queries=[vector_query],
                vector_filter_mode=VectorFilterMode.PRE_FILTER,
                filter = filter_expression,  
                top=top_k,
                
            )
            search_results = list(search_results)

            # Step 4: Process and return results with blob URLs
            results = []
            for result in search_results:
                doc = {
                    "content": result["content"],
                    "filename": result["filename"],
                    "page_number": result["page_number"],
                    "score": result["@search.score"],
                    "download_url": self.get_blob_url(result["filename"]),
                    "view_url": f"/view_pdf/{result['filename']}",
                }
                results.append(doc)
            self.logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            raise
    

    