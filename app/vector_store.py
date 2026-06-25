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
    def __init__(self, persist_path: str= chroma_persists_path, embed_model:str= 'sentence-transformers/all-mpnet-base-v2'):
        self.chroma = Chroma(
            persist_directory=persist_path,
            embedding_function=HuggingFaceEmbeddings(model_name=embed_model) if embed_model else None,
            collection_name='agora_documents'
        )

    def search(self, query: str, strategy: str='similarity')->Optional[list[Document]]:
        if strategy == 'similarity':
            return self.chroma.similarity_search(
                query=query,
                k=3
            )
        elif strategy == 'mmr':
            return self.chroma.max_marginal_relevance_search(
                query=query,
                k=5,
                fetch_k=20,
                lambda_mult=0.7
            )
        else:
            None