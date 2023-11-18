from environs import Env
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

env = Env()
env.read_env()
KEY = env.str("KEY")
OPENAI_API_BASE = env.str("OPENAI_API_BASE")

llm = OpenAI(openai_api_key=KEY, openai_api_base=OPENAI_API_BASE)

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# CommaSeparatedListOutputParser().parse("hi, bye")

chat_model = ChatOpenAI(openai_api_key=KEY, openai_api_base=OPENAI_API_BASE)

# predict: 接受一个字符串，返回一个字符串

# print(llm.predict("hi"))

# print(chat_model.predict("hi"))
# text = "What would be a good company name for a company that makes colorful socks?"
# print(llm.predict(text))

# predict_messages: 接受一个消息列表，返回一个消息。
# text = "制造多彩袜子的公司的好名字是什么？"
# messages = [HumanMessage(content=text)]
# result = llm.predict_messages(messages, temperature=0.5)

# prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
# prompt.format(product="colorful scocks")
# print(prompt.template)

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)

result = chain.run("colors")
print(result)