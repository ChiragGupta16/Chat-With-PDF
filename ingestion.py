import langchain
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb

dir = "db"
loader = PDFMinerLoader("Freight_report.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=150)
text = text_splitter.split_documents(data)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = Chroma.from_documents(text,embeddings,persist_directory=dir)
db.persist()
db=None 
