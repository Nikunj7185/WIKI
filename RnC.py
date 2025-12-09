from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def RequestAndChunk(topics: list, embedding):
    docs = WikipediaLoader(
        query=topics, 
        load_max_docs=len(topics),
        doc_content_chars_max=10000
    ).load()

    print(f"Total initial documents loaded: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(docs)
    lib = FAISS.from_documents(chunks, embedding) 
    
    return lib