import gc
from itertools import islice

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from preprocessing import prepare_segment_docs, configure_text_splitter, prepare_segment_summary_docs, \
    prepare_long_summary_document_docs, prepare_short_summary_document_docs

'''

injection.py is a dedicated codebase to configure db and inject documents.


'''

def get_embeddings_strategy(model:str):
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

def pull_all_documents():
    text_splitter = configure_text_splitter(chunk_size=1200, overlap_size=120)
    segments = prepare_segment_docs(text_splitter)
    segment_summaries = prepare_segment_summary_docs(text_splitter)
    long_summaries = prepare_long_summary_document_docs(text_splitter)
    short_summaries = prepare_short_summary_document_docs()
    documents = []
    documents.extend(segments)
    documents.extend(segment_summaries)
    documents.extend(long_summaries)
    documents.extend(short_summaries)
    return documents

def get_documents_generator(text_splitter):
    yield from prepare_segment_docs(text_splitter)
    yield from prepare_segment_summary_docs(text_splitter)
    yield from prepare_long_summary_document_docs(text_splitter)
    yield from prepare_short_summary_document_docs()

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
    embedding_model = get_embeddings_strategy("sentence-transformers/all-mpnet-base-v2")

    print("Configure vector store...")
    vectorstore = configure_vector_store(embedding_model, db='chroma', collection_name='agora_documents', path='assets/db/chroma_1')

    print("Configure document generators...")
    text_splitter = configure_text_splitter(chunk_size=1200, overlap_size=120)
    documents_generator = get_documents_generator(text_splitter)

    print("Injecting documents...")
    batch_size = 20
    for batch in batch_iterator(documents_generator, batch_size):
        vectorstore.add_documents(documents=batch)
        del batch
        gc.collect()

    print("Finish injection")

