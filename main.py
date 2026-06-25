from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import pydantic 

load_dotenv()

system_promt ="""
#身份
- 你是一个ai编程助手，擅长用企业级的代码编写程序

#指令
- 用snake case命名，不要用camel case命名法
-  不要返回markdwn格式说明，只需要返回代码

#示例


"""

agent = create_agent(
    model="deepseek-chat",
    system_prompt=system_promt,
)

human_messages = HumanMessage(content="用python写一个简洁的贪心算法")

messages = agent.stream(
    {"messages":human_messages},
    stream_mode="messages",
)
for token,metadata in messages:
    if token:
        print(token.content,end="",flush=True)