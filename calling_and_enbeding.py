import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
model_name = os.environ.get("MODEL_NAME")

if not api_key:
    print("ERROR: No OPENAI_API_KEY found. Check your .env file.")
    exit()

DOC_PATH = "knowlage.txt"
DB_DIR = "chroma_db"

def load_document (path):
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()

def split_documant(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def get_embeding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vector_store(embedding_model):
    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model
        )

    docs = load_document(DOC_PATH)
    chunks = split_documant(docs)
    print(f"Created {len(chunks)} chunks. Embedding and storing...")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

def get_llm():
    return ChatOpenAI(
        model= model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0
    )

prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant.
     Use the folllowing context to answer the question.
     if the answer is not in the context, say you don't know.
     
     Context : {context}
     
     Question : {question}

     Give a concise and factual answer. 
     """
)

def format_docs(docs):
    return "\n\n" .join(doc.page_content for doc in docs)


def build_cahin(retriever, llm):
    return(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

if __name__== "__main__":
    print("Loading embedding model (first run downloads it )...")

    embedding_model = get_embeding_model()
    vector_store = get_vector_store(embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()
    rag_chain = build_cahin(retriever, llm)

    print("\n==== RAG Assistant ====")
    print("Ask about the document. Type 'exit' to quit.\n")

    while True:
        question = input("YOU : ").strip()

        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye")
            break

        if not question:
            print("Please type something.\n")
            continue

        try:
            answer = rag_chain.invoke(question)
            print(f"AI : {answer}")
        except Exception as e:
            print("Something went wrong: ", e, "\n")
