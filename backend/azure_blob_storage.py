import traceback

from backend.utility import Utility


class AzureBlobStorage(Utility):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()
        self.AZURE_BLOB_CONTAINER_NAME = self._get_env_variables("AZURE_BLOB_CONTAINER_NAME")

    def __initialize_services(self):
        """
        Initialise Azure Blob client
        Return : AzureBlobServiceClient
        """
        from azure.storage.blob import BlobServiceClient

        AZURE_STORAGE_CONNECTION_STRING = self._get_env_variables("AZURE_STORAGE_CONNECTION_STRING")
        try:
            self.AZURE_BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(
                AZURE_STORAGE_CONNECTION_STRING
            )
            self.logger.info("AzureBlobServiceClient initalized successfully")
        except Exception as e:
            self.logger.error("AzureBlobServiceClient initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

    # def upload_pdf_to_blob(self, file_path: str, blob_name: str=None, from_ui: bool = False, meta_data=None) -> str:
    #     """
    #     Upload a PDF file to Azure Blob Storage

    #     Args:
    #         file_path: Local path to the PDF file
    #         blob_name: Name to give the blob in storage

    #     Returns:
    #         URL of the uploaded blob
    #     """
    #     print(f"Uploading PDF to blob storage: {file_path} as {blob_name}")
    #     try:
    #         if from_ui and meta_data is None:
    #             raise Exception("meta_data is required from ui.")
    #         if blob_name is None and not from_ui:
    #             raise Exception("blob_name is required.")

    #         if from_ui:
    #             from datetime import datetime
    #             filename = meta_data.get("filename", "")
    #             project_code = meta_data.get("project_code", "")
    #             label_tag = meta_data.get("label_tag", "")

    #             # Create blob name with custom path structure
    #             blob_name = (
    #                 f"Categories/{project_code}/{project_code}_{filename}"
    #             )
    #             # print(f"Blob name generated: {blob_name}")
    #             # Upload blob with metadata
    #             blob_metadata = {
    #                 "filename": filename,
    #                 "project_code": project_code,
    #                 "label_tag": label_tag,
    #                 "upload_timestamp": datetime.now().isoformat(),
    #             }
    #         else:
    #             blob_metadata = {}
    #             # if meta_data is None:
    #             #     blob_metadata = {}
    #             # else:
    #             #     blob_metadata = meta_data

    #         # Get a reference to the blob client
    #         blob_client = self.AZURE_BLOB_SERVICE_CLIENT.get_blob_client(
    #             container=self.AZURE_BLOB_CONTAINER_NAME,
    #             blob=blob_name
    #         )

    #         # Upload the file to blob storage
    #         with open(file_path, "rb") as data:
    #             blob_client.upload_blob(data, overwrite=True, metadata=blob_metadata)

    #         self.logger.info(f"PDF uploaded to blob storage: {blob_name}")
    #         return blob_name

    #     except Exception as e:
    #         self.logger.error(f"Error uploading PDF to blob storage: {str(e)}")
    #         self.logger.error(traceback.format_exc())
    #         raise e

    def upload_pdf_to_blob(self, file_path: str, blob_name: str=None, from_ui: bool = False, meta_data=None) -> str:
        """
        Upload a PDF file to Azure Blob Storage

        Args:
            file_path: Local path to the PDF file
            blob_name: Name to give the blob in storage

        Returns:
            URL of the uploaded blob
        """
        try:
            if from_ui and meta_data is None:
                raise Exception("meta_data is required from ui.")
            if blob_name is None and not from_ui:
                raise Exception("blob_name is required.")

            if from_ui:
                from datetime import datetime
                filename = meta_data.get("filename", "")
                project_code = meta_data.get("project_code", "")
                label_tag = meta_data.get("label_tag", "")

                # Create blob name with custom path structure
                if ".pdf" in filename:
                    if project_code is filename:
                        blob_name = (
                            f"Categories/{project_code}/{project_code}_{filename}"
                        )
                    else:
                        blob_name = (
                            f"Categories/{project_code}/{filename}"
                        )
                else:
                    if project_code is filename:
                        blob_name = (
                            f"Categories/{project_code}/{filename}.pdf"
                        )
                    else:
                        blob_name = (
                            f"Categories/{project_code}/{project_code}_{filename}.pdf"
                        )
                # Upload blob with metadata
                blob_metadata = {
                    "filename": filename,
                    "project_code": project_code,
                    "label_tag": label_tag,
                    "upload_timestamp": datetime.now().isoformat(),
                }
            else:
                blob_metadata = {}
                # if meta_data is None:
                #     blob_metadata = {}
                # else:
                #     blob_metadata = meta_data

            # Get a reference to the blob client
            blob_client = self.AZURE_BLOB_SERVICE_CLIENT.get_blob_client(
                container=self.AZURE_BLOB_CONTAINER_NAME,
                blob=blob_name
            )

            # Upload the file to blob storage
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, metadata=blob_metadata)

            self.logger.info(f"PDF uploaded to blob storage: {blob_name}")
            return blob_name, blob_client.url

        except Exception as e:
            self.logger.error(f"Error uploading PDF to blob storage: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e

    def get_azure_blob_client(self, blob_name: str):
        return self.AZURE_BLOB_SERVICE_CLIENT.get_blob_client(
                container=self.AZURE_BLOB_CONTAINER_NAME, blob=blob_name
            )

    def get_blob_url(self, blob_name: str) -> str:
        return self.get_azure_blob_client(blob_name=blob_name).url

    def get_pdf_content_from_blob(self, blob_name: str):
        blob_client = self.get_azure_blob_client(blob_name=blob_name)
        return blob_client.download_blob().readall()

