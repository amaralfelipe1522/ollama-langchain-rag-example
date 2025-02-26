# ollama-langchain-rag-example
Exemplo de comunicação básica necessária para estruturar um RAG local utilizando Ollama, LangChain e FAISS.

## Pré-Requisitos
Instalar o [Ollama](https://ollama.com/) e baixar os modelos de IA necessários.

## TO DO

Uso de PDFs

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("documento.pdf")
```