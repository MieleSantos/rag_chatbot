import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def temp_file_save(file):
    """
        Recebe um arquivo em binario e salva no diretorio temporario
    Args:
        file (_type_): Arquivo binario do pdf

    Returns:
        _type_: path do arquivo gerado
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
        return temp_file_path


def process_pdf(file):
    path_file = temp_file_save(file)

    loader = PyPDFLoader(path_file)
    docs = loader.load()

    os.remove(path_file)
    return create_chunks(docs)


def create_chunks(docs):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = text_spliter.split_documents(documents=docs)
    return chunks


def add_to_vector_store(chunks, persist_directory, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
    return vector_store


def load_existing_vector_store(persist_directory):
    # conectando no banco caso ele já exista
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
        )
        return vector_store
    return None
