#import streamlit as st
import os
from langchain_groq import ChatGroq
#from langchain_community.vectorstores import Chroma
#from langchain_community.chat_models import ChatOllama
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.prompts import ChatPromptTemplate
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains import create_retrieval_chain

GROQ_API_KEY = os.getenv("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")

if not GROQ_API_KEY:
    st.error("Por favor, configure a variável de ambiente GROQ_API_KEY.")
    st.stop()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192" # O modelo Llama 3 8B no Groq
)

# --- MODIFICAÇÃO CRUCIAL PARA DOCKER ---
# Aponte para o serviço 'ollama' no docker-compose na porta 11434
llm = ChatOllama(model="llama3:8b", base_url="http://ollama:11434")

# Carregar a função de embedding e o banco de dados
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Template do Prompt
template = """
Você é um Mestre de Jogo assistente para Dungeons & Dragons.
Sua especialidade é explicar regras de forma clara e dar exemplos práticos.
Responda a pergunta do usuário baseando-se SOMENTE no contexto das regras fornecido.
Se a informação não estiver no contexto, diga que não encontrou a regra específica no material disponível.

Contexto:
{context}

Pergunta: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# Criação das cadeias (chains) do LangChain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Interface do Streamlit
st.title("🧙‍♂️ Sou Adélio Mago dos livros")
st.subheader("Seu assistente pessoal de regras")

question = st.text_input("Faça sua pergunta sobre as regras de D&D:")

if question:
    with st.spinner("Consultando os tomos antigos..."):
        # Verifica se o banco de dados existe antes de tentar a busca
        if not os.path.exists("./chroma_db"):
            st.error("O banco de dados de vetores não foi encontrado! Você já executou o script 'ingest.py'?")
        else:
            response = rag_chain.invoke({"input": question})
            st.write(response["answer"])