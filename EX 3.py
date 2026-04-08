import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv(".env")

model = ChatOpenAI(
    model=os.getenv("AZURE_MODEL_NAME"),
    temperature=0.1,
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT")
)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

model_with_tools = model.bind_tools([get_word_length])

response = model_with_tools.invoke(
    "How many letters are in the word 'hello'?"
)

print("\nFinal Answer:", response.content)