import os
import streamlit as st

# from agent import select_model
from process_data import add_to_vector_store, load_existing_vector_store, process_pdf,select_model
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY')
persist_directory = 'database/db'
vector_store = load_existing_vector_store(persist_directory)

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
                chunks = process_pdf(file=upload_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                persist_directory,chunks=all_chunks,vector_store=vector_store
            )

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM', options=select_model()
    )


question = st.chat_input('como posso ajuda?')
st.chat_message('user').write(question)
st.chat_message('ai').write('ia')
