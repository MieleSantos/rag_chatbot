import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFRepository:
    @classmethod
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

    def load_pdf(cls, file):
        path_file = cls.temp_file_save(file)

        loader = PyPDFLoader(path_file)
        docs = loader.load()

        os.remove(path_file)
        return docs
        # return CreateChunks.create_chunks(docs)


class CreateChunks:
    @classmethod
    def create_chunks(docs):
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=400
        )
        chunks = text_spliter.split_documents(documents=docs)
        return chunks
