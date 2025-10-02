from openai import OpenAI

from src.settings import settings

def create_openai_client():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    return client

class llm_connection():
    def __init__(self):
        self.client = create_openai_client()
        self.prompt = ""

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_system_message(self, system_message: str = "You are a helpful assistant."):
        self.system_message = system_message

    def generate_prompt_response(self) -> str|None:
        client = self.client
        response = client.chat.completions.create(
            model=settings.openrouter_model_id,
            messages=[
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": f"{self.prompt}"
                }
            ]
            )
        return response.choices[0].message.content
    
    def generate_formatted_response(self, messages, response_format):
        """
        Generate a response from the LLM with a specified response format.
        Can only be used by certain models that support response_format.
        https://openrouter.ai/docs/features/structured-outputs
        
        """
        client = self.client
        response = client.chat.completions.create(
            model=settings.openrouter_model_id,
            messages=messages,
            response_format=response_format
            )

        return response
    
