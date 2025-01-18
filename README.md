
# Chatbot com OpenAI, LangChain, Poetry e Streamlit usando RAG

Este é um projeto de chatbot desenvolvido com a técnica de Geração com Recuperação Aumentada (RAG), integrando OpenAI, LangChain, Poetry e Streamlit. O objetivo é fornecer respostas mais precisas e relevantes ao usuário ao combinar um sistema de recuperação de informações com geração de linguagem natural.

## Tabela de Conteúdos

- [Descrição](#descrição)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Contribuições](#contribuições)


## Descrição

Este chatbot usa a técnica de RAG, onde respostas geradas são aprimoradas através da recuperação de informações pertinentes ao contexto do usuário. Ao fazer perguntas, o modelo primeiro consulta uma base de conhecimento e, em seguida, combina as informações recuperadas com uma geração de resposta natural, permitindo respostas precisas e contextuais.

## Tecnologias Utilizadas

- **OpenAI**: Utilizado para gerar respostas de linguagem natural.
- **LangChain**: Facilita o processo de integração e pipeline de recuperação e geração de conteúdo.
- **Poetry**: Gerenciador de dependências e ambiente virtual para Python.
- **Streamlit**: Interface web para uma experiência interativa com o chatbot.
- **RAG (Retrieval-Augmented Generation)**: 

## RAG (Retrieval-Augmented Generation)

A técnica de Geração com Recuperação Aumentada (RAG, ou Retrieval-Augmented Generation) é uma abordagem que combina recuperação de informações com geração de linguagem natural para melhorar a precisão e relevância das respostas de um modelo de IA.

Em vez de apenas gerar respostas com base em uma pergunta, o modelo RAG primeiro consulta uma base de conhecimento ou um conjunto de documentos para recuperar informações relevantes. Em seguida, ele usa essas informações recuperadas como contexto para gerar uma resposta mais detalhada e informada. Isso é especialmente útil para questões complexas ou técnicas, onde a precisão e a riqueza de detalhes são fundamentais.

A técnica é estruturada em dois passos principais:

**Recuperação**: Um módulo de busca localiza documentos ou trechos de informações relevantes na base de conhecimento.

**Geração**: O modelo de geração (como GPT, da OpenAI) usa essas informações para criar uma resposta que é precisa e diretamente ligada ao contexto recuperado.

O RAG é amplamente utilizado em chatbots avançados e sistemas de suporte ao cliente, pois permite que a IA forneça respostas específicas e atualizadas sem a necessidade de ter todo o conteúdo previamente memorizado.

## Instalação

1. **Clone este repositório:**

   ```bash
   git clone https://github.com/seu_usuario/seu_projeto.git
   cd seu_projeto
   ```

2. **Instale as dependências com o Poetry:**

   Certifique-se de ter o Poetry instalado. Se não, instale com:

   ```bash
   pip install poetry
   ```

   Depois, instale as dependências:

   ```bash
   poetry install
   ```

3. **Configuração da API do OpenAI:**

   Você precisará de uma chave de API da OpenAI. Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Como Usar

1. **Inicie o ambiente virtual com o Poetry:**

   ```bash
   poetry shell
   ```

2. **Execute o chatbot com o Streamlit:**

   ```bash
   streamlit run app.py
   ```

   ou 

    ```bash
   task run
   ```

3. **Interaja com o chatbot:**  
   Acesse `http://localhost:8501` no seu navegador e comece a interagir com o chatbot. Ele responderá com base nas informações recuperadas e na geração aumentada.<br>
   Para fazer RAG, escolha um arquivo pdf com os dados e faça perguntas para o chatbot sobre o arquivo



## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas e pull requests para melhorias no projeto.


