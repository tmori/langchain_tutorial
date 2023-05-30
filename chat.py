import sys 

doc_dir='./documents'
db_dir='DB'
llm_name="gpt-3.5-turbo"
embedding_model='text-embedding-ada-002'
chank_size = 100

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

def create_db(doc_dir, db_dir, llm_name, embedding_model, chank_size):
    pdf_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pdf")]

    pages = []
    for pdf_file in pdf_files:
        print("PDFロード：" + pdf_file)
        loader = PyPDFLoader(doc_dir + '/' + pdf_file)
        # PDF ドキュメントの内容を分割する
        print("PDF ドキュメントの内容を分割する")
        tmp_pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",  # セパレータ
            chunk_size = chank_size,  # チャンクの文字数
            chunk_overlap = 0,  # チャンクオーバーラップの文字数
        )
        chanked_pages = text_splitter.split_documents(tmp_pages)
        # 連結
        pages = pages + chanked_pages

    # LangChain における LLM のセットアップ
    print("LangChain における LLM のセットアップ")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name=llm_name, max_tokens=1000)  

    # 分割したテキストの情報をベクターストアに格納する
    print("分割したテキストの情報をベクターストアに格納する")
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    #embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()

    # PDF ドキュメントへ自然言語で問い合わせる
    pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    return pdf_qa

def load_db(db_dir, llm_name, embedding_model):
    # LangChain における LLM のセットアップ
    print("LangChain における LLM のセットアップ")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name=llm_name, max_tokens=1000)  

    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    #embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    return pdf_qa

def do_chat(pdf_qa):
    chat_history = []
    while True:
        query = input("> ")
        print("質問：" + query)

        result = pdf_qa({"question": query, "chat_history": chat_history})

        print("回答："+result["answer"])


if mode == "new":
    pdf_qa = create_db(doc_dir, db_dir, llm_name, embedding_model, chank_size)
    sys.exit(0)
else:
    pdf_qa = load_db(db_dir, llm_name, embedding_model)
    do_chat(pdf_qa)
    sys.exit(0)

