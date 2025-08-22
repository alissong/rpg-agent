import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Definir o caminho para a pasta de PDFs e para o banco de dados
PDF_DIRECTORY = "./pdfs"
PERSIST_DIRECTORY = "./chroma_db"

def get_pdf_files():
    """Encontra todos os arquivos .pdf na pasta de PDFs."""
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    return pdf_files

def create_documents_from_pdfs(pdf_files):
    """Carrega PDFs, divide em páginas e adiciona metadados."""
    all_docs = []
    for pdf_file in pdf_files:
        try:
            print(f"Processando livro: {pdf_file}...")
            loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, pdf_file))
            docs = loader.load()
            
            # Adiciona o nome do arquivo como metadado 'source' em cada página.
            # Isso é CRUCIAL para saber de qual livro veio a informação.
            for doc in docs:
                doc.metadata["source"] = pdf_file
            
            all_docs.extend(docs)
            print(f"'{pdf_file}' carregado com sucesso. {len(docs)} páginas processadas.")
        except Exception as e:
            print(f"Erro ao carregar o arquivo {pdf_file}: {e}")
    return all_docs

def main():
    """Função principal para criar o banco de dados vetorial."""
    
    # Encontra os arquivos PDF
    pdf_list = get_pdf_files()
    if not pdf_list:
        print(f"Nenhum arquivo PDF encontrado na pasta '{PDF_DIRECTORY}'.")
        print("Por favor, adicione seus livros em PDF e tente novamente.")
        return

    print("Iniciando processo de ingestão de dados...")
    
    # Carrega e processa os documentos
    documents = create_documents_from_pdfs(pdf_list)
    if not documents:
        print("Nenhum documento foi carregado. Encerrando.")
        return
        
    print("\nDividindo os documentos em pedaços (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    print("Criando embeddings (isso pode levar um tempo na primeira vez)...")
    # Usa um modelo de embedding local e gratuito
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Criando e salvando o banco de dados em '{PERSIST_DIRECTORY}'...")
    # Cria e salva o banco de dados Chroma localmente
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)

    print("\n--- Processo Concluído! ---")
    print(f"Seu banco de dados foi criado com {len(splits)} chunks de texto de {len(pdf_list)} livros.")
    print("Agora você pode iniciar a aplicação principal.")

if __name__ == "__main__":
    main()