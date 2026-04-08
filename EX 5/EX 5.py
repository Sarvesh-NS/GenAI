import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("AZURE_MODEL_NAME"),
    temperature=0.1,
    max_tokens=1000,
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT"),
)

@tool
def get_word_info(word: str) -> str:
    """Returns technical info about a word."""
    length = len(word)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
    return f"The word '{word}' has {length} letters and {vowels} vowels."


researcher_prompt = SystemMessage(
    content="You are a Researcher. Use tools to analyze words."
)

researcher_agent = create_tool_calling_agent(
    model=model,
    tools=[get_word_info],
    system_message=researcher_prompt
)

researcher = AgentExecutor(
    agent=researcher_agent,
    tools=[get_word_info],
    verbose=True
)
writer_prompt = SystemMessage(
    content="You are a poet. Convert given facts into a beautiful poem."
)
writer_agent = create_tool_calling_agent(
    model=model,
    tools=[],
    system_message=writer_prompt
)
writer = AgentExecutor(
    agent=writer_agent,
    tools=[],
    verbose=True
)
word = "hello"
print("\n🔍 Research Phase\n")
facts = researcher.invoke({
    "messages": [HumanMessage(content=f"Analyze the word '{word}'")]
})
print("\nWriting Phase\n")
poem = writer.invoke({
    "messages": [
        HumanMessage(content=f"{facts['output']}\nWrite a poem from this.")
    ]
})
print("\nFINAL OUTPUT:\n")
print(poem["output"])