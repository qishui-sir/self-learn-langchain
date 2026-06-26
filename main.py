from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

system_promt ="""
#身份
- 你是一个ai编程助手，擅长用企业级的代码编写程序

#指令
- 用snake case命名，不要用camel case命名法
-  不要返回markdwn格式说明，只需要返回代码

#示例


"""

Search_Tavily = TavilySearch(
    max_result = 5,
    topic = "general",
)

agent = create_agent(
    model="deepseek-chat",
    system_prompt=system_promt,
    tools=[Search_Tavily],
    checkpointer=InMemorySaver(),
)

config = {"configurable":{"thread_id":"1"}}

human_messages = HumanMessage(content="我最喜欢猫咪")
response = agent.invoke(
    {"messages":human_messages},
    config,
)

human_messages = HumanMessage(content="你知道我最喜欢什么动物吗")
response = agent.invoke(
    {"messages":human_messages},
    config,
)
print(response["messages"][-1].content+"\n")

def Search_Tavly(query: str):
    """Search web for the imformation"""
    return Search_Tavily.invoke(query)