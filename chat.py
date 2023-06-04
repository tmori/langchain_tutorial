import sys

llm_name="gpt-4"
embedding_model='text-embedding-ada-002'
page_chunk_size = 1024
max_token_num = 4096
conversation_window_size = 3
conversation_token_num = 1024
conversation_history_type = "token" # token or window
inference_phase_num = 1
#print("arg num=" + str(len(sys.argv)))

if (len(sys.argv) == 1) or (len(sys.argv) > 4):
    print("USAGE: " + sys.argv[0] + " new [<doc_dir> [<db_dir>]]")
    print("USAGE: " + sys.argv[0] + " chat [<db_dir>]")
    sys.exit(1)

mode=sys.argv[1]
db_dir = "DB"
doc_dir = "documents"

if mode == "chat":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("USAGE: " + sys.argv[0] + " chat [<db_dir>]")
        sys.exit(1)
    if len(sys.argv) == 3:
        db_dir = sys.argv[2]
    

if mode == "new":
    if len(sys.argv) != 2 and len(sys.argv) != 4:
        print("USAGE: " + sys.argv[0] + " new [<doc_dir> [<db_dir>]]")
        sys.exit(1)
    if len(sys.argv) == 4:
        doc_dir=sys.argv[2]
        db_dir = sys.argv[3]

print("DB_DIR =" + db_dir)
print("DOC_DIR=" + doc_dir)

import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory

def create_db(doc_dir, db_dir, embedding_model, chunk_size, token_num):
    pdf_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pdf")]
    csv_files = [ file for file in os.listdir(doc_dir) if file.endswith(".csv")]
    pptx_files = [ file for file in os.listdir(doc_dir) if file.endswith(".pptx")]
    url_files = [ file for file in os.listdir(doc_dir) if file.endswith(".url")]
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap = 0,
    )
    files = pdf_files + csv_files + pptx_files + url_files
    pages = []
    for file in files:
        print("INFO: Loading document=" + file)
        if ".pdf" in file:
            loader = PyPDFLoader(doc_dir + '/' + file)
        elif ".csv" in file:
            loader = CSVLoader(doc_dir + '/' + file)
        elif ".pptx" in file:
            loader = UnstructuredPowerPointLoader(doc_dir + '/' + file)
        elif ".url" in file:
            with open(doc_dir + '/' + file, 'r') as file:
                urls = file.read().splitlines()
            loader = UnstructuredURLLoader(urls = urls)
        else:
            print("WARNING: Not supported document=" + file)
            continue
        print("INFO: Spliting document=" + file)
        tmp_pages = loader.load_and_split()
        chanked_pages = text_splitter.split_documents(tmp_pages)
        pages = pages + chanked_pages

    print("INFO: Storing Vector DB")
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()


def load_db(db_dir, llm_name, embedding_model, token_num, history_type, num):
    print("INFO: Setting up LLM")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0, 
        model_name=llm_name, 
        max_tokens=token_num)

    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    if (history_type == "window"):
        memory = ConversationBufferWindowMemory(k=num, memory_key="chat_history", return_messages=True)
    else:
        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=num, memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        memory=memory
        )
    return pdf_qa

def load_db_with_type(db_dir):
    global llm_name, max_token_num, conversation_history_type, conversation_window_size, conversation_token_num
    if (conversation_history_type == "window"):
        pdf_qa = load_db(db_dir, llm_name, embedding_model, max_token_num, conversation_history_type, conversation_window_size)
    else:
        pdf_qa = load_db(db_dir, llm_name, embedding_model, max_token_num, conversation_history_type, conversation_token_num)
    return pdf_qa

def do_chat(db_dir):
    pdf_qa = load_db_with_type(db_dir)
    while True:
        query = input("> ")
        if query == 'exit' or query == 'q' or query == "quit":
            print("See you again!")
            sys.exit(0)
        print("Q: " + query)

        result = pdf_qa({"question": query})

        print("A: "+result["answer"])

def do_chat2(db_dir):
    while True:
        pdf_qa = load_db_with_type(db_dir + "/summary")
        query = input("> ")
        if query == 'exit' or query == 'q' or query == "quit":
            print("See you again!")
            sys.exit(0)
        first_query = "「" + query + "」という質問文に対して、その質問に回答可能な情報が多いと思われるドキュメントを必ず１個抽出し、以下の書式で出力してください。RESULT: DocumentName"
        print("Q: " + first_query)
        result = pdf_qa({"question": first_query})
        print("A: "+result["answer"])
        input_string=result["answer"]
        tmp = input_string.split(":")
        db_name = tmp[len(tmp) - 1].strip()
        print("INFO: selected db_name: "+db_name)
        pdf_qa = load_db_with_type(db_dir + "/" + db_name)
        result = pdf_qa({"question": query})
        print("Q: " + query)
        print("A: "+result["answer"])

if mode == "new":
    pdf_qa = create_db(doc_dir, db_dir, embedding_model, page_chunk_size, max_token_num)
else:
    if inference_phase_num == 1:
        do_chat(db_dir)
    else:
        do_chat2(db_dir)

sys.exit(0)
