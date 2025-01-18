import os
import tempfile

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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


def add_to_vector_store(persist_directory, chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
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


def select_model():
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]

    return model_options


# def prompt(context):
#     system_prompt='''
#         Use o contexto para responder as perguntas.
#         Se não encontrar uma resposta no contexto,
#         explique que não há informações disponiveis.
#         Responda em formtato de mmarkdown e com visualizações
#         elaboradas e interativas.
#         Contexto: {context}
#     '''
def ask_question(model, query, vector_store, st):
    llm = ChatOpenAI(model=model)

    retriever = vector_store.as_retriever()

    system_prompt = """
        Use o contexto para responder as perguntas.
        Se não encontrar uma resposta no contexto,
        explique que não há informações disponiveis.
        Responda em formtato de mmarkdown e com visualizações
        elaboradas e interativas.
        Contexto: {context}
    """
    # primeira mensagem
    messages = [('system', system_prompt)]
    # historico ded mensagens
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    # ultima mensagem
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )

    response = chain.invoke({'input': query})
    return response.get('answer'), st
