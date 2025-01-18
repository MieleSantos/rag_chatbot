import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from  repository.database.config import get_database

db = get_database()

class VectorRepository:
    @classmethod
    def add_to_vector_store(self,chunks, vector_store=None):
        if vector_store:
            vector_store.add_documents(chunks)
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                persist_directory=db,
                embedding_function=OpenAIEmbeddings(),
            )
        return vector_store

    @classmethod
    def load_existing_vector_store(self):
        # conectando no banco caso ele já exista
        
        if os.path.exists(os.path.join(db)):
            vector_store = Chroma(
                persist_directory=db,
                embedding_function=OpenAIEmbeddings(),
            )
            return vector_store
        return None
