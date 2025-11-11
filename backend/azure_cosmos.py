import traceback
import pandas as pd

from backend.utility import Utility

class AzureCosmos(Utility):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()
        self.initialize_cosmosdb()

    def __initialize_services(self):
        from azure.cosmos import CosmosClient

        AZURE_COSMOS_DB_URI = self._get_env_variables("AZURE_COSMOS_DB_URI")
        AZURE_COSMOS_DB_KEY = self._get_env_variables("AZURE_COSMOS_DB_KEY")
        AZURE_COSMOS_DB_DATABASE_NAME = self._get_env_variables("AZURE_COSMOS_DB_DATABASE_NAME")
        self.AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER = self._get_env_variables("AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER")

        if not AZURE_COSMOS_DB_URI:
            raise Exception("AZURE_COSMOS_DB_URI environment variable not set")
        if not AZURE_COSMOS_DB_KEY:
            raise Exception("AZURE_COSMOS_DB_KEY environment variable not set")
        if not AZURE_COSMOS_DB_DATABASE_NAME:
            raise Exception("AZURE_COSMOS_DB_DATABASE_NAME environment variable not set")

        try:
            self.AZURE_COSMO_DB_CLIENT = CosmosClient(
                AZURE_COSMOS_DB_URI,
                {"masterKey": AZURE_COSMOS_DB_KEY}
            )
            self.logger.info("AZURE_COSMO_DB_CLIENT initalized successfully")
        except Exception as e:
            self.logger.error("AZURE_COSMO_DB_CLIENT initalization failed")
            self.logger.error(traceback.format_exc())
            raise e

        try:
            self.AZURE_COSMO_DB = self.AZURE_COSMO_DB_CLIENT.get_database_client(AZURE_COSMOS_DB_DATABASE_NAME)
            self.logger.info("AZURE_COSMO_DB initialized successfully")
        except Exception as e:
            self.logger.error("AZURE_COSMO_DB initialized failed")
            self.logger.error(traceback.format_exc())
            raise e

    def initialize_cosmosdb(self) -> None:
        """
        This function is used to initialize the azure cosmos db client and establish the connection to it

        Args:
            container_name (str): The name of the container to be initialized. If None, a ValueError will be raised.

        Returns:
            None
        """
        from azure.cosmos import exceptions
        try:
            if self.AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER is not None:
                self.AZURE_COSMO_DB_CONTAINER = self.AZURE_COSMO_DB.get_container_client(self.AZURE_COSMOS_DB_CHAT_HISTORY_CONTAINER)
                if not self.AZURE_COSMO_DB_CONTAINER:
                    print("[ERROR] Cosmos DB container not initialized")
            else:
                print("[ERROR] container_name is None")
                raise ValueError("container_name is None")
        except exceptions.CosmosResourceNotFoundError as e:
            print("[ERROR] CosmosResourceNotFoundError: ", e)
        except Exception as e:
            print("[ERROR] initialize_cosmosdb(): ", e)

    def get_cosmo_query(self, question_column, user_id, session_id, file_name=None, project_code=None):
        if not file_name or not project_code:
            cosmos_query = f"""SELECT TOP 3 c.{question_column}, c.rephrased_question, c.answer FROM chat_sessions c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' ORDER BY c._ts DESC"""
        else:
            cosmos_query = f"""SELECT TOP 3 c.{question_column}, c.rephrased_question, c.answer FROM chat_sessions c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.project_code = '{project_code}' AND c.file_name = '{file_name}' ORDER BY c._ts DESC"""
        return cosmos_query

    def read_cosmo_table(self, question_column, user_id, session_id, file_name=None, project_code=None) -> pd.DataFrame:
        """
        This function is used to read the dataframe from Azure Cosmos DB based on the provided SQL query and container name.

        Args:
            query (str): The SQL query to read the data from the container.
            container_name (str): The name of the container to read the data from.

        Returns:
            pd.DataFrame: The dataframe of the data read from Azure Cosmos DB.
            None: If no results are found.
            False: If an error occurs during the reading process.
        """

        try:
            query = self.get_cosmo_query(question_column, user_id, session_id, file_name, project_code)
            if self.AZURE_COSMO_DB_CONTAINER is None:
                print("[ERROR] Cosmos DB container not initialized")
            items = self.AZURE_COSMO_DB_CONTAINER.query_items(query=query, enable_cross_partition_query=True)
            results = list(items)
            if not results:
                return None
            return pd.DataFrame(results)
        except Exception as e:
            print("[ERROR] read_table(): ", e)
            return False

    # Front-End Function
    def get_cosmo_user_chat_history(self, user_id, limit=50) -> list:
        """Get chat history for a user"""
        if self.AZURE_COSMO_DB_CONTAINER is None:
            return []

        try:
            query = (f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.type = 'chat_message' "
                     f"ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}")
            items = list(
                self.AZURE_COSMO_DB_CONTAINER.query_items(query=query, enable_cross_partition_query=True)
            )
            return items
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []

    def get_cosmo_user_sessions_message(self, user_id, session_id):
        if self.AZURE_COSMO_DB_CONTAINER is None:
            return []
        try:
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.type = 'chat_message' ORDER BY c.timestamp ASC"
            items = list(
                self.AZURE_COSMO_DB_CONTAINER.query_items(query=query, enable_cross_partition_query=True)
            )
            return items
        except Exception as e:
            print(f"Error getting user sessions messages : {e}")
            return []

    def get_cosmo_user_sessions(self, user_id):
        """Get all sessions for a user"""
        if self.AZURE_COSMO_DB_CONTAINER is None:
            return []

        try:
            query = f"SELECT DISTINCT c.session_id FROM c WHERE c.user_id = '{user_id}' AND c.type = 'chat_message' ORDER BY c.timestamp DESC "
            items = list(
                self.AZURE_COSMO_DB_CONTAINER.query_items(query=query, enable_cross_partition_query=True)
            )
            for idx, session in enumerate(items):
                try:
                    items_2 = self.get_cosmo_user_sessions_message(user_id=user_id, session_id=session['session_id'])[0]
                    items[idx]["question"] = items_2["question"]
                except:
                    items[idx]["question"] = "Unknown Question"
            return items
        except Exception as e:
            print(f"Error getting user sessions: {e}")
            return []

    def save_cosmo_chat_message(
            self,
            user_id,
            conversation_id,
            session_id,
            question,
            answer,
            timestamp,
            rephrased_question,
            retrieved_documents,
            source_documents=None,
    ):
        import uuid
        """Save chat message to Cosmos DB"""
        if self.AZURE_COSMO_DB_CONTAINER is None:
            return []

        try:
            chat_message = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "question": question,
                "rephrased_question": rephrased_question,
                "answer": answer,
                "timestamp": timestamp,
                "source_documents": source_documents or [],
                "retrieved_documents": retrieved_documents,
                "type": "chat_message",
            }

            self.AZURE_COSMO_DB_CONTAINER.create_item(chat_message)
            return True
        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False

    def delete_cosmo_chat_message(self, user_id, session_id):
        if self.AZURE_COSMO_DB_CONTAINER is None:
            return False
        try:
            # Get all messages for this session
            query = f"SELECT c.id FROM c WHERE c.user_id = '{user_id}' AND c.session_id = '{session_id}' AND c.type = 'chat_message'"
            items = list(
                self.AZURE_COSMO_DB_CONTAINER.query_items(query=query, enable_cross_partition_query=True)
            )
            for item in items:
                self.AZURE_COSMO_DB_CONTAINER.delete_item(item["id"], partition_key=user_id)
            return True
        except Exception as e:
            print(f"Error deleting chat message: {e}")
            return False
