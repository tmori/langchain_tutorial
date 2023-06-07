from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools import Tool
from langchain.tools.base import ToolException
from langchain.memory import ConversationBufferMemory
from chat import load_db_with_type
import re
import os

db_dir = "../dbs/all"
llm_name="gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=llm_name, temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
def _handle_error(error:ToolException) -> str:
    return  "The following errors occurred during tool execution:" + error.args[0]+ "Please try another tool."

def search_tool(query: str):
    global db_dir
    #print(query)
    qa = load_db_with_type(db_dir)
    result = qa({"question": query})
    #print("A: "+result["answer"])
    return result["answer"]


def translation_en_tool(s: str):
    global llm
    query = "Please translate following sentences in English: " + s
    print(query)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(tools = [], 
                               llm = llm, 
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                               memory=memory,
                               verbose=True)
    ret = agent_chain.run(input=query)
    print("結果：" + ret)
    return ret

def translation_jp_tool(s: str):
    global llm
    query = "Please translate following sentences in Japanease: " + s
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(tools = [], 
                               llm = llm, 
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                               memory=memory,
                               verbose=True)
    ret = agent_chain.run(input=query)
    return ret

tools = [     
    Tool.from_function(
        func=search_tool,
        name="Search_tool",
        description="When answering the question, it is helpful to extract information from the document that holds the relevant information.",
        handle_tool_error=_handle_error,
    ),
    Tool.from_function(
        func=translation_en_tool,
        name="Translation_from_Japanease_to_English_tool",
        description="useful for when you need to Translation from Japanease to English.",
        handle_tool_error=_handle_error,
    ),
    Tool.from_function(
        func=translation_jp_tool,
        name="Translation_from_English_to_Japanease_tool",
        description="useful for when you need to Translation from English to Japanease.",
        handle_tool_error=_handle_error,
    ),
]

template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            #print("action.log=" + action.log)
            #print("observation=" + observation)
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        msg = HumanMessage(content=formatted)
        return [msg]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print("LLM_OUTPUT:" + llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()


# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

while True:
    q = input("> ")
    if q == 'exit' or q == 'q' or q == "quit":
        print("See you again!")
        break
    #query = translation_en_tool(q)
    query = q
    print("Q: " + query)
    ret_msg = agent_executor.run(query)
    ret_msg = translation_jp_tool(ret_msg)
    print("A:" + ret_msg)

print(memory.load_memory_variables({}))
