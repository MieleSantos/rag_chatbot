import os

import streamlit as st

from agent import select_model, ask_question_vector
from repository.process_data import CreateChunks, PDFRepository
from repository.vector_repository import VectorRepository

vector_store = VectorRepository.load_existing_vector_store()

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

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('como posso ajuda?')
# st.chat_message('user').write(question)
# st.chat_message('ai').write('ia')

if question and vector_store:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    print(selected_model)
    with st.spinner('Buscando resposta...'):
        response, st = ask_question_vector.search_assistent(
            model=selected_model, query=question, vector_store=vector_store, st=st
        )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})
