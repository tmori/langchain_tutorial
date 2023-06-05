import sys
import re
import csv
import pandas as pd

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
ans_dir = "answer"

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

def create_db(doc_dir, db_dir, embedding_model, chunk_size):
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

    print("INFO: Storing Vector DB:" + db_dir)
    embeddings = OpenAIEmbeddings(deployment=embedding_model)
    vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()


def load_db(db_dir, llm_name, embedding_model, token_num, history_type, num):
    print("INFO: Setting up LLM:" + db_dir)
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

def get_question():
    ret_query = ""
    query = input("Input Question> ")
    if query == 'exit' or query == 'q' or query == "quit":
        print("See you again!")
        sys.exit(0)
    if len(query.strip()) == 0:
        return ""
    ret_query = "質問:" + query
    query_background = input("Input BackGround> ")
    if query_background == 'exit' or query_background == 'q' or query_background == "quit":
        print("See you again!")
        sys.exit(0)
    if len(query_background.strip()) > 0:
        ret_query = ret_query + " 背景:" + query_background

    concrete_question = input("Input Concrete Question> ")
    if query_background == 'exit' or query_background == 'q' or query_background == "quit":
        print("See you again!")
        sys.exit(0)
    if len(concrete_question.strip()) > 0:
        ret_query = ret_query + " 具体的な質問:" + concrete_question
    return ret_query

def check_question(pdf_qa, question):
    #query = "「" + question + "」という質問文に対して、回答作成する上で必要な背景情報やより質問を具体化した方が良ければ、ERROR: <理由> という書式で理由を示してください。問題なければ、OKと回答ください"
    query = "For the question '" + question + "', if there is a need for additional background information or further specification to properly answer, please indicate the reason using the format ERROR: <reason>. If there's no issue, please respond with 'OK'."
    print("Q: " + query)
    result = pdf_qa({"question": query})
    print("A: "+result["answer"])
    if "ERROR" in result["answer"]:
        return False
    return True


def check_existing_data(question, filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if question in df.iloc[:,0].values:
            return True
        else:
            return False
    else:
        return False

def save_to_csv(string1, string2, filename):
    # 文字列から改行コードを除去
    string1 = string1.replace('\n', '')
    string2 = string2.replace('\n', '')
    is_exist = os.path.exists(filename)

    # CSVファイルにデータを書き込む
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        if is_exist == False:
            writer.writerow(['Question', 'Answer'])  # ヘッダー行
        writer.writerow([string1, string2])

def append_best_answer(ans_dir, db_name, query, answer):
    global embedding_model, page_chunk_size
    file_path = ans_dir + "/" + db_name + ".csv"
    if check_existing_data(query, file_path) == False:
        save_to_csv(query, answer, file_path)
        _ = create_db(ans_dir, ans_dir + "/DB", embedding_model, page_chunk_size)

def try_best_answer(question):
    ans_db_path = "answer/DB"
    if os.path.exists(ans_db_path) == False:
        return False
    qa = load_db_with_type(ans_db_path)
    query = "Please provide the answer, appropriately breaking into new lines, if there is a response to the question corresponding to '" + question + "'. If not found, please indicate the reason using the format ERROR: <reason>."
    print("Q: " + query)
    result = qa({"question": query})
    print("A: "+result["answer"])
    if "ERROR" in result["answer"]:
        print("WARNING: NOT FOUND BEST ANSWER")
        return False
    print("INFO: FOUND BEST ANSWER")
    return True

def do_chat2(db_dir):
    global ans_dir
    while True:
        pdf_qa = load_db_with_type(db_dir + "/summary")
        query = get_question()
        if not query:
            print("ERROR: Invalid question...")
            continue
        if check_question(pdf_qa, query) == False:
            continue
        if try_best_answer(query) == True:
            continue
        first_query = "For the question '" + query + "', please identify a document that seems to contain ample information to answer it, and provide the response in the following format: RESULT: <DocumentName>"
        print("Q: " + first_query)
        result = pdf_qa({"question": first_query})
        print("A: "+result["answer"])
        input_string=result["answer"]
        tmp = input_string.split(":")
        db_name = tmp[len(tmp) - 1].strip()
        match = re.search(r"([a-zA-Z0-9_-]+)$", db_name)
        if match:
            db_name = match.group(1)
        else:
            print("ERROR: can not find best document...")
            continue
        print("INFO: selected db_name: "+db_name)
        pdf_qa = load_db_with_type(db_dir + "/" + db_name)
        result = pdf_qa({"question": query})
        print("Q: " + query)
        print("A: "+result["answer"])
        append_best_answer(ans_dir, db_name, query, result["answer"])

if mode == "new":
    pdf_qa = create_db(doc_dir, db_dir, embedding_model, page_chunk_size)
else:
    if inference_phase_num == 1:
        do_chat(db_dir)
    else:
        do_chat2(db_dir)

sys.exit(0)
