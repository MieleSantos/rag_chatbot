import streamlit as st

from agent import select_model
from database.config import get_database
from repository.process_data import CreateChunks, PDFRepository
from repository.vector_repository import VectorRepository

vector_store = VectorRepository.load_existing_vector_store(get_database())

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.header('🤖 Chat com seus documentos (RAG)')

with st.sidebar:
    st.header('Upload de arquivo')
    upload_files = st.file_uploader(
        label='Faça o upload de arquivos PDF', type=['pdf'], accept_multiple_files=True
    )
    if upload_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for upload_file in upload_files:
                docs = PDFRepository.load_pdf(file=upload_file)
                chunks = CreateChunks.create_chunks(docs)
                all_chunks.extend(chunks)
            vector_store = VectorRepository.add_to_vector_store(
                chunks=all_chunks, vector_store=vector_store
            )

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM', options=select_model()
    )


question = st.chat_input('como posso ajuda?')
st.chat_message('user').write(question)
st.chat_message('ai').write('ia')
