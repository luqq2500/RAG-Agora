import os.path
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

base_path = Path(__file__).resolve().parent.parent
chroma_persists_path = base_path / "dev" / "assets" / "db" / "chroma"

class VectorStore(ABC):
    @abstractmethod
    def search(self, query: str, strategy: str)->Optional[list[Document]]:
        pass

class ChromaDB(VectorStore):
    def __init__(self, persist_path: str= chroma_persists_path, collection_name: str = 'agora_documents', embed_model:str= 'sentence-transformers/all-mpnet-base-v2'):
        if not os.path.exists(chroma_persists_path):
            raise FileNotFoundError(f"File to persistent paths not found.")

        try:
            self.chroma = Chroma(
                persist_directory=persist_path,
                embedding_function=HuggingFaceEmbeddings(model_name=embed_model),
                collection_name=collection_name,
            )

            collection = self.chroma._client.get_collection(name=collection_name)
            count = collection.count()
            if count == 0:
                raise ValueError(f"Collection {collection_name} is empty.")

        except Exception as e:
            raise RuntimeError(f'Failed to initialize chroma: {e}')

    def search(self, query: str, strategy: str='similarity')->Optional[list[Document]]:
        if strategy == 'similarity':
            return self.chroma.similarity_search(query=query,k=3)
        elif strategy == 'mmr':
            return self.chroma.max_marginal_relevance_search(query=query, k=5, fetch_k=20, lambda_mult=0.7)
        else:
            None