import traceback
from typing import List

from backend.utility import Utility


class AzureOpenAI(Utility):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)
        self.__initialize_services()

    def __initialize_services(self):
        from openai import OpenAI, AzureOpenAI

        self.use_azure_openai = self._get_env_variables("USE_AZURE_OPENAI", "false")
        self.use_azure_openai = self.use_azure_openai.lower() == "true"

        if self.use_azure_openai:
            try:
                AZURE_OPENAI_ENDPOINT = self._get_env_variables("AZURE_OPENAI_ENDPOINT")
                AZURE_OPENAI_API_KEY = self._get_env_variables("AZURE_OPENAI_API_KEY")
                AZURE_OPENAI_API_VERSION = self._get_env_variables(
                    "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
                )
                self.AZURE_OPENAI_CHAT_DEPLOYMENT = self._get_env_variables("AZURE_OPENAI_CHAT_DEPLOYMENT")
                self.EMBEDDING_DEPLOYMENT = self._get_env_variables("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                self.OPENAI_CLIENT = AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
                self.logger.info("Azure OPENAI_CLIENT initalized successfully")
            except Exception as e:
                self.logger.error("Azure OPENAI_CLIENT initalization failed")
                self.logger.error(traceback.format_exc())
                raise e
        else:
            try:
                OPENAI_API_KEY = self._get_env_variables("OPENAI_API_KEY")
                self.EMBEDDING_DEPLOYMENT = self._get_env_variables("OPENAI_EMBEDDING_MODEL")
                self.OPENAI_CHAT_MODEL = self._get_env_variables("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
                self.OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, max_retries=3)
                self.logger.info("OPENAI_CLIENT initalized successfully")
            except Exception as e:
                self.logger.error("OPENAI_CLIENT initalization failed")
                self.logger.error(traceback.format_exc())
                raise e

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's Ada model
        Supports both standard OpenAI and Azure OpenAI

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            if self.use_azure_openai:

                # For Azure OpenAI, we need to use the deployment name as the model
                response = self.OPENAI_CLIENT.embeddings.create(
                    model=self.EMBEDDING_DEPLOYMENT, input=texts
                )
            else:
                # For standard OpenAI, use the model name
                response = self.OPENAI_CLIENT.embeddings.create(
                    model=self.EMBEDDING_DEPLOYMENT, input=texts
                )

            # Extract embedding vectors from the response
            embeddings = [embedding.embedding for embedding in response.data]
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def get_openai_response(self, messages, temperature=0.1, json_object: bool = True):
        if json_object:
            if self.use_azure_openai:
                # For Azure OpenAI, use the deployment name as the model
                response = self.OPENAI_CLIENT.chat.completions.create(
                    model=self.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    messages=messages,

                    temperature=temperature,  # Low temperature for more focused answers
                    response_format={
                        "type": "json_object",
                    },
                )
            else:
                # For standard OpenAI, use the model name
                response = self.OPENAI_CLIENT.chat.completions.create(
                    model=self.OPENAI_CHAT_MODEL,
                    messages=messages,
                    # max_tokens=max_tokens,
                    temperature=temperature,  # Low temperature for more focused answers
                    response_format={
                        "type": "json_object",
                    },
                )
        else:
            if self.use_azure_openai:
                # For Azure OpenAI, use the deployment name as the model
                response = self.OPENAI_CLIENT.chat.completions.create(
                    model=self.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    messages=messages,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                # For standard OpenAI, use the model name
                response = self.OPENAI_CLIENT.chat.completions.create(
                    model=self.OPENAI_CHAT_MODEL,
                    messages=messages,
                    # max_tokens=max_tokens,
                    temperature=temperature,  # Low temperature for more focused answers
                )
        response = response.choices[0].message.content
        return response
