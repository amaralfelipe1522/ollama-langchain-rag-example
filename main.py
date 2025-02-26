import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Criar embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Verificar se o conteúdo da RAG já existe no FAISS
if os.path.exists("meu_banco_faiss"):
    print("Conteúdo encontrado...")
    vectorstore = FAISS.load_local("meu_banco_faiss", embeddings, allow_dangerous_deserialization=True)
    print("Conteúdo Carregado!")
else:
    # Carregar e criar os chunks
    loader = TextLoader("meu_conteudo.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Indexar com FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("meu_banco_faiss")

# Criar uma cadeia de recuperação e geração
retriever = vectorstore.as_retriever()
print("Cadeia de recuperação e geração finalizada")

llm = Ollama(model="deepseek-r1")

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")

# Tipos de chain_type
# "stuff"	Junta tudo e envia para o modelo responder. Simples, mas pode estourar o limite de tokens.
# "map_reduce"	Divide os documentos, gera respostas para cada um e depois sintetiza a resposta final.
# "refine"	O modelo gera uma resposta inicial e vai refinando conforme recebe mais contexto.
# "map_rerank"	Gera múltiplas respostas e escolhe a melhor baseada em relevância.

# Consultar a IA
query = "Me responda brevemente em PT-BR. Qual o nome da empresa?"
response = qa_chain.invoke(query)

print(response["result"])