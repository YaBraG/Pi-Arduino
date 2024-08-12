'''
                Working Ollama
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import AIMessage
from sentence_transformers import SentenceTransformer


llm = Ollama(model="llama3")
llm = ChatOllama(model = "llama3",temperature=0,#other parameters...
                )
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    Document(page_content="Rene likes Sucking Dicks.",metadata={"id":0})
]

vector_store = Chroma.from_documents(documents,embedding=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

queries = input("Enter Prompt: ")


response = qa_chain.run(queries)
print(f"Query: {queries}\nResponse : {response}\n")

# message = [("system","You are a helpful assistant that translates English to French. Translate the user sentence."),("human","I love the pink")]
# ai_msg = llm.invoke(message)
# print(ai_msg)
'''

'''
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = UnstructuredFileLoader("ai_adoption_framework_whitepaper.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)

llm = Ollama(model="llama3")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

question = "Can you please summarize the document"
result = chain.invoke({"query": question})

print(result['result'])
'''

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./ai_adoption_framework_whitepaper.pdf",extract_images=True)
pages = loader.load()
print(pages)