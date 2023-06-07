from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#from getpass import getpass
import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
tools = []

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
agent_chain = initialize_agent(tools, 
                               llm, 
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                               verbose=True, 
                               memory=memory)

while True:
    arg = input("> ")
    if arg == "q" or arg == "quit":
        print("See you again!")
        break
    ret = agent_chain.run(input=arg)
    print(ret)


print(memory.load_memory_variables({}))