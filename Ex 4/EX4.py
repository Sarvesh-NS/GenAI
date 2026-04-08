import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT"),
    model=os.getenv("AZURE_MODEL_NAME"),
    temperature=0.1
)

@tool
def get_word_info(word: str) -> str:
    length = len(word)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
    return f"The word '{word}' has {length} letters and {vowels} vowels."

research_agent = initialize_agent(
    tools=[get_word_info],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

def writer(facts):
    prompt = f"You are a poet. Write a beautiful poem using:\n{facts}"
    return llm.invoke(prompt).content

word = "Pranesh"
print(f"\nChecking word: {word}\n")

facts = research_agent.run(f"Get info about '{word}'")
print("Researcher Output:")
print(facts)

poem = writer(facts)
print("\nWriter Output:\n")
print(poem)