import openai
import requests

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class OpenAI:
    def __init__(self, openai_api_key, engine="gpt-4", use_requests=False):
        self.api_key = openai_api_key
        self.model = engine
        self.base_url = "https://api.openai.com/v1/engines/chat/completions"
        self.generate = self.generate_with_requests if use_requests else self.generate_with_openai

        # Prepare OpenAI API package
        openai.api_key = openai_api_key

    def generate_with_openai(self, prompt, system=DEFAULT_SYSTEM_MESSAGE, **model_kwargs):
        response = openai.ChatCompletion.create(
            engine=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **model_kwargs
        )
        return response["choices"][0]["message"]["content"]

    def generate_with_requests(self, prompt, system=DEFAULT_SYSTEM_MESSAGE, **model_kwargs):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        }
        data.update(model_kwargs)
        response = requests.post(self.base_url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]


class AzureOpenAI(OpenAI):
    def __init__(self, azure_config, model, API_VERSION= "2023-07-01-preview", use_requests=True):
        super().__init__(azure_config.get("OPENAI_API_KEY"), engine=azure_config.get(model), use_requests=use_requests)
        self.resource = azure_config.get("RESOURCE")
        self.model = azure_config.get(model)
        self.base_url = f"https://{self.resource}.openai.azure.com/openai/deployments/{self.model}/chat/completions?api-version={API_VERSION}"

        # Prepare OpenAI API package
        openai.api_base = self.base_url
        openai.api_type = "azure"
        openai.api_version = API_VERSION
