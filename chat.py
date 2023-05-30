import sys 

doc_dir='./documents'
db_dir='DB'
llm_name="gpt-3.5-turbo"
embedding_model='text-embedding-ada-002'
page_chunk_size = 256
max_token_num = 2048
conversation_window_size = 3

if len(sys.argv) != 2:
    print("ERROR: " + sys.argv[0] + " <new|load>")
    sys.exit(1)

mode=sys.argv[1]

import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

def create_db(doc_dir, db_dir, embedding_model, chunk_size, token_num):
    pdf_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pdf")]

    pages = []
    for pdf_file in pdf_files:
        print("PDFロード：" + pdf_file)
        loader = PyPDFLoader(doc_dir + '/' + pdf_file)
        # PDF ドキュメントの内容を分割する
        print("PDF ドキュメントの内容を分割する")
        tmp_pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(
            separator = "\n",  # セパレータ
            chunk_size = chunk_size,  # チャンクの文字数
            chunk_overlap = 0,  # チャンクオーバーラップの文字数
        )
        chanked_pages = text_splitter.split_documents(tmp_pages)
        # 連結
        pages = pages + chanked_pages

    # 分割したテキストの情報をベクターストアに格納する
    print("分割したテキストの情報をベクターストアに格納する")
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    #embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()


def load_db(db_dir, llm_name, embedding_model, token_num, window_num):
    # LangChain における LLM のセットアップ
    print("LangChain における LLM のセットアップ")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0, 
        model_name=llm_name, 
        max_tokens=token_num)

    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    #embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    memory = ConversationBufferWindowMemory(k=window_num, memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        memory=memory
        )
    return pdf_qa

def do_chat(pdf_qa):
    while True:
        query = input("> ")
        print("質問：" + query)

        result = pdf_qa({"question": query})

        print("回答："+result["answer"])


if mode == "new":
    pdf_qa = create_db(doc_dir, db_dir, embedding_model, page_chunk_size, max_token_num)
    sys.exit(0)
else:
    pdf_qa = load_db(db_dir, llm_name, embedding_model, max_token_num, conversation_window_size)
    do_chat(pdf_qa)
    sys.exit(0)

