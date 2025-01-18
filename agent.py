import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

from database.config import get_database


class ChatbotRag:
    def __init__(self, modelo: str):
        load_dotenv()
        self._setup_environment_variables()
        self.model = ChatOpenAI(model=modelo)
        self.prompt_template = self._create_prompt_templats()
        self.db = get_database()
        self.toolkit = self._toolkit()
        self.system_message = self._system_message()
        self.agent_executor = self._create_agent_executor()

    def _setup_environment_variables(self):  # noqa: PLR6301
        api_key = os.getenv('API_KEY')

        if not api_key:
            raise ValueError(
                "A chave da API 'API_KEY' não foi encontrada nas variáveis de ambiente."
            )
        os.environ['OPENAI_API_KEY'] = api_key

    def _create_prompt_template(self):  # noqa: PLR6301
        prompt = """
        Use as ferramentas necessárias para responder perguntas relacionadas ao
        estoque de produtos. Você fornecerá insights sobre produtos, preços,
        reposição de estoque e relatórios conforme solicitado pelo usuário.
        A resposta final deve ter uma formatação amigável de visualização para usuário.
        Sempre responda em português brasileiro.
        Pergunta: {q}
        """
        return PromptTemplate.from_template(prompt)

    def _toolkit(self):
        return SQLDatabaseToolkit(db=self.db, llm=self.model)

    def _system_message(self):  # noqa: PLR6301
        return hub.pull('hwchase17/react')

    def _create_agent_executor(self):
        agent = create_react_agent(
            llm=self.model, tools=self.toolkit.get_tools(), prompt=self.system_message
        )
        return AgentExecutor(agent=agent, tools=self.toolkit.get_tools(), verbose=True)

    def search_assistent(self, question: str) -> str:  # noqa: PLR6301
        if not question:
            raise ValueError('A pergunta não pode ser vazia')

        input_question = self.prompt_template.format(q=question)
        response = self.agent_executor.invoke({'input': input_question})

        return response  # .get('output')


def select_model():  # noqa: PLR6301
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]

    return model_options


def chatbot_query(question, modelo):
    assistent = ChatbotRag(modelo=modelo)
    return assistent.search_assistent(question)
