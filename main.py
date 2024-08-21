import json
import traceback

from typing import List
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.llms import Tongyi
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_wenxin.chat_models import ChatWenxin

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--api_key', default=None, type=str, help='large language model api_key')
    parser.add_argument('--document_path', default=".\data", type=str, help='number of total epochs to run')
    parser.add_argument('--document_name', default="slotcraft.pdf", type=str, help='mini-batch size (default: 16)')

    config = parser.parse_args()
    return config

def split_document(file_path):
    documents = PyPDFLoader(file_path).load()
    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)

    return split_docs

class QaPair(BaseModel):
    instruction: str = Field(description='指令')
    input: str = Field(description='输入')
    output: str = Field(description='输出')

class QaPairs(BaseModel):
    qas: List[QaPair] = Field(description='问答对列表')


QA_PAIRS_SYSTEM_PROMPT = """  
<Context></Context> 标记中是一段文本，学习和分析它，并整理学习成果：  
- 提出问题并给出每个问题的答案。  
- 答案需详细完整，尽可能保留原文描述。  
- 答案可以包含普通文字、链接、代码、表格、公式等。  
- 问题和答案都必须为中文。  
- 问题能提的越多越好,尽量超过15个。
- 问题能问的越详细越好，答案也需要尽可能的详细。  
"""

QA_PAIRS_HUMAN_PROMPT = """  
请严格按以下JSON格式整理学习成果,任何多余的东西都不要有，任何必要的东西都不能少:
[  
{{"instruction": "指令1","input":"输入1","output":"输出1"}},  
{{"instruction": "指令2","input":"输入2","output":"输出2"}}
]  

------
我给你举个例子，严格按照例子模版输出结果:
[
    {{
        "instruction": "根据给定的坐标确定最近的机场。",
        "input": "40.728157, -73.794853",
        "output": "距给定坐标最近的机场是纽约市的拉瓜迪亚机场 (LGA)。"
    }},
    {{
        "instruction": "输出不同种类水果的列表",
        "input": "",
        "output": "1.苹果 2.香蕉 3.橘子 4.芒果 5.草莓 6.葡萄 7. 蓝莓 8.樱桃 9.猕猴桃 10.甜瓜 11.菠萝 12.李子 13.桃子"
    }},
    {{
        "instruction": "找出字串中隐藏的信息",
        "input": "业余咖啡",
        "output": "隐藏的消息是“咖啡因爱好者”。"
    }}
]

------  
我们开始吧!  

<Context>  
{text}  
<Context/>  
"""


def create_chain(
        api_key: str
):
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_PAIRS_SYSTEM_PROMPT),
        ("human", QA_PAIRS_HUMAN_PROMPT)
    ])
    # 智谱
    zhipuai_api_key = api_key
    llm = ChatZhipuAI(
        temperature=0,
        api_key=zhipuai_api_key,
        model="glm-4"
    )
    parser = JsonOutputParser(pydantic_object=QaPairs)
    chain = prompt | llm | parser

    # chain = prompt | llm
    return chain


def main():

    config = vars(parse_args())

    chain = create_chain(config['api_key'])
    documents = split_document(config["document_path"]+config["document_name"])

    bar = tqdm(total=len(documents))
    for idx, doc in enumerate(documents):
        document = doc.copy()
        if idx == 0:
            document.page_content = doc.page_content + documents[idx+1].page_content
        elif idx == len(documents)-1:
            document.page_content = documents[idx-1].page_content + doc.page_content
        else:
            document.page_content = documents[idx-1].page_content + doc.page_content + documents[idx+1].page_content

        try:
            out = chain.invoke({'text': document.page_content})
        except:
            cc = "pdf的第" + str(idx) + "页有问题，请重新检查\n"
            print(cc)
            with open('error_log.txt', 'a', encoding='utf-8') as f_err:
                f_err.write(cc)
                f_err.write("Exception occurred:\n")
                traceback.print_exc(file=f_err)
                f_err.write("\n")
                out.content = "\n"

        if not isinstance(out, str):
            out = entities.content
        print(out)
        with open(f'dataset.txt', 'a', encoding='utf-8') as f:
            f.write(out)
        bar.update(1)

    bar.close()
if __name__ == '__main__':
    main()
