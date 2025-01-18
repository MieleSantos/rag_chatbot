import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorRepository:
    @classmethod
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

    @classmethod
    def load_existing_vector_store(persist_directory):
        # conectando no banco caso ele já exista
        if os.path.exists(os.path.join(persist_directory)):
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings(),
            )
            return vector_store
        return None
