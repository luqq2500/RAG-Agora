from itertools import islice
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from preprocessing import stream_segment_docs, configure_text_splitter, stream_segment_summary_docs, \
    stream_long_summary_docs, stream_short_summary_docs

'''

injection.py is a dedicated codebase to:
    1. Create vector database.
    2. Stream preprocessed documents to add to vector database.
    3. Use 'Generator' to stream documents, where methods 'stream_*_docs' used as sub-generator for generator 'batch_iterator()'

'''

def get_embeddings(model:str):
    if model == "openai-text-embedding-3-small":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif model == 'sentence-transformers/all-mpnet-base-v2':
        return HuggingFaceEmbeddings(model_name=model)
    else:
        return None

def configure_vector_store(text_embedding_model, db='chroma', collection_name='agora_documents', path="assets/db/chromadb"):
    if db=='chroma':
        return Chroma(
            collection_name=collection_name,
            embedding_function=text_embedding_model,
            persist_directory=path,
        )
    else:
        return None

def stream_docs_generators(text_splitter):
    yield from stream_segment_docs(text_splitter)
    yield from stream_segment_summary_docs(text_splitter)
    yield from stream_long_summary_docs(text_splitter)
    yield from stream_short_summary_docs()

def batch_iterator(iterable, size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch

if __name__ == "__main__":

    print("Initiate injection...")

    print("Getting embedding model...")
    embedding_model = get_embeddings("sentence-transformers/all-mpnet-base-v2")

    print("Configure vector store...")
    vectorstore = configure_vector_store(embedding_model, db='chroma', collection_name='agora_documents', path='assets/db/chroma_1')

    print("Configure document generators...")
    text_splitter = configure_text_splitter(chunk_size=1200, overlap_size=120)

    print("Injecting documents...")
    batch_size = 10
    for batch in batch_iterator(stream_docs_generators(text_splitter), batch_size):
        vectorstore.add_documents(documents=batch)
        del batch

    print("Finish injection")

