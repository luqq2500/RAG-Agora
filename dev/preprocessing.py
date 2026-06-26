import os
import pathlib
from dataclasses import asdict
from typing import Any, Generator

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pandas import Series

from domain import TextSplitMetadata, DocumentMetadata

'''
preprocessing.py codebase is dedicated for preprocessing AGORA raw text entities:
1. Document: Segment, segment summary, long summary, short summary
2. Collection: Description

# Preprocessing strategy applied:
1. Macro context injection: document overview, usecase policy.
2. Chunking: chunk large texts
3. Metadata: document, usecase policy, chunk indices

# Key methods:
    1. segment_docs_generator(): Stream segment Documents.
    2. segment_summary_docs_generator(): Stream 
    3. long_summary_docs_generator(): To prepare document long summary Documents.
    4. short_summary_docs_generator(): To prepare document short summary Documents.

'''

def stream_collection_desc_docs():
    df = pd.read_csv(collections_path)
    for index, row in df.iterrows():
        content = row.get('Description')
        collection_context = row.get('Name')
        metadata = {'collection': collection_context}
        yield Document(page_content=content, metadata=metadata)

def stream_segment_summary_docs(text_splitter: RecursiveCharacterTextSplitter)->Generator[Document, None, None]:
    df = merge_segment_document_authorities()
    for index, row in df.iterrows():
        content = row.get("Summary")
        if pd.isna(content):
            continue
        overview = prepare_overview_context(row)

        usecase_flags = collect_usecase_flags(row)
        usecases = prepare_usecase_context(usecase_flags)

        segment_portion = f'{str(row.get("Segment position"))}/{str(row.get("Number of segments created"))}'
        segment_context = f'Segment: {segment_portion}'
        segment_metadata = {'segment': f'{segment_portion}'}

        document_metadata = prepare_document_metadata(row)

        text_chunks = text_splitter.split_text(content) if content.strip() else [""]
        for chunk_idx, text_chunk in enumerate(text_chunks):
            chunk_metadata, chunk_context = prepare_chunk_metadata(chunk_idx, len(text_chunks), include_context=True)
            chunk_content = f'{chunk_context} {overview} {usecases} {segment_context}\n\n---\n\n{text_chunk}'.strip()
            metadata = {**document_metadata, **chunk_metadata, **segment_metadata}
            yield Document(page_content=chunk_content, metadata=metadata)

def stream_segment_docs(text_splitter: RecursiveCharacterTextSplitter)->Generator[Document, None, None]:
    df = merge_segment_document_authorities()
    for index, row in df.iterrows():
        content = row.get("Text")
        if pd.isna(content):
            continue

        overview = prepare_overview_context(row)
        usecase_flags = collect_usecase_flags(row)
        usecases = prepare_usecase_context(usecase_flags)

        segment_portion = f'{str(row.get("Segment position"))}/{str(row.get("Number of segments created"))}'
        segment_context = f'Segment: {segment_portion}'
        segment_metadata = {'segment': f'{segment_portion}'}
        document_metadata = prepare_document_metadata(row)

        text_chunks = text_splitter.split_text(content) if content.strip() else [""]
        for chunk_idx, text_chunk in enumerate(text_chunks):
            chunk_metadata, chunk_context = prepare_chunk_metadata(chunk_idx, len(text_chunks), include_context=True)
            chunk_content = f'{chunk_context} {overview} {usecases} {segment_context}\n\n---\n\n{text_chunk}'.strip()
            metadata = {**document_metadata, **chunk_metadata, **segment_metadata}
            yield Document(page_content=chunk_content, metadata=metadata)

def stream_long_summary_docs(text_splitter: RecursiveCharacterTextSplitter, target_column='Long summary')->Generator[Document, None, None]:
    df = merge_document_authorities()
    for index, row in df.iterrows():
        content = row.get(target_column)
        if pd.isna(content):
            continue
        document_metadata = prepare_document_metadata(row)

        text_chunks = text_splitter.split_text(content)
        for chunk_idx, text_chunk in enumerate(text_chunks):
            chunk_metadata, chunk_header = prepare_chunk_metadata(chunk_idx, len(text_chunks), include_context=True)
            chunk_content = f'{chunk_header}\n\n---\n\n{text_chunk}'.strip()
            metadata = {**document_metadata, **chunk_metadata}
            yield Document(page_content=chunk_content, metadata=metadata)

def stream_short_summary_docs()->Generator[Document, None, None]:
    df = merge_document_authorities()
    for index, row in df.iterrows():
        content = row.get("Short summary")
        if pd.isna(content):
            continue
        metadata = prepare_document_metadata(row)
        yield Document(page_content=content, metadata=metadata)

def configure_text_splitter(chunk_size, overlap_size, mode='recursive-character-text-splitter'):
    if mode == 'recursive-character-text-splitter':
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    else:
        return None

def prepare_chunk_metadata(chunk_index: int, total_chunks: int, include_context: bool):
    metadata = TextSplitMetadata(current_chunk=chunk_index+1, total_chunks=total_chunks)
    if include_context:
        return asdict(metadata), f'Chunk: {chunk_index+1}/{total_chunks}'
    else:
        return asdict(metadata)

def prepare_overview_context(row: Series) -> str:
    agora_id = row.get('AGORA ID', "an unidentified AGORA ID")
    official_name = row.get('Official name', "an undisclosed official title")
    source = row.get('Link to document', 'an undisclosed source')
    authority = row.get('Authority', "an unnamed authority")
    jurisdiction = row.get('Jurisdiction', 'an unnamed jurisdiction')
    collection = row.get('Collections', "an unnamed collection")
    proposed_on = row.get('Proposed date', 'an undisclosed proposed date')
    recent_activity = row.get('Most recent activity', 'an undisclosed recent activity')
    recent_activity_date = row.get('Most recent activity date', 'an undisclosed recent activity date')
    overview = (
        f"Document Title: '{official_name}' | AGORA ID: '{agora_id}' | "
        f"Issuing authority: '{authority}' | Jurisdiction: '{jurisdiction}' | "
        f"Dates: Proposed '{proposed_on}' | Recent Activity: '{recent_activity}' on '{recent_activity_date}' | "
        f"Source Collection: '{collection}' | URL: '{source}' ")
    return overview

def prepare_usecase_context(usecase_flags: dict[str, list[str]]):
    semantics = []
    if usecases := usecase_flags.get('applications'):
        semantics.append(f"Applications: '{', '.join(usecases)}'.")
    if usecases := usecase_flags.get('risks'):
        semantics.append(f"Risk Factors: '{', '.join(usecases)}'.")
    if usecases := usecase_flags.get('harms'):
        semantics.append(f"Harms Mitigation: '{', '.join(usecases)}'.")
    if usecases := usecase_flags.get('strategies'):
        semantics.append(f"Regulatory Strategies: '{', '.join(usecases)}'.")
    if usecases := usecase_flags.get('incentives'):
        semantics.append(f"Enforcement Mechanisms: '{', '.join(usecases)}'.")
    if usecase := usecase_flags.get('sectors'):
        semantics.append(f"Target Sectors: '{', '.join(usecase)}'.")
    return " ".join(semantics) if semantics else "Policy Usecases: 'None'"

def prepare_document_metadata(row: Series) -> dict[str, Any]:
    usecase_flags = collect_usecase_flags(row)
    usecase_flattened = {key: ", ".join(value) if value is not None else None for key, value in usecase_flags.items()}

    metadata = DocumentMetadata(
        agora_id=row.get('AGORA ID'),
        authority=row.get('Authority'),
        collection=row.get('Collections'),
        status=row.get('Most recent activity'),
        applications=usecase_flattened['applications'],
        harms=usecase_flattened['harms'],
        incentives=usecase_flattened['incentives'],
        risks=usecase_flattened['risks'],
        strategies=usecase_flattened['strategies'],
        sectors=usecase_flattened['sectors']
    )

    return {key: value for key, value in asdict(metadata).items() if value is not None} # omit None field from document metadata

def collect_usecase_flags(row: Series) -> dict[str, list[str]]:
    applications = []
    harms = []
    incentives = []
    risks = []
    strategies = []
    sectors = []
    for col, value in row.items():
        if value is not True or str(value).lower() != 'true':
            continue

        col_str = str(col).strip()
        if col_str.startswith('Applications: '):
            applications.append(col_str.replace("Applications: ", "").strip())
        elif col_str.startswith('Harms: '):
            harms.append(col_str.replace("Harms: ", "").strip())
        elif col_str.startswith('Incentives: '):
            incentives.append(col_str.replace("Incentives: ", "").strip())
        elif col_str.startswith('Risk factors: '):
            risks.append(col_str.replace("Risk factors: ", "").strip())
        elif col_str.startswith('Strategies: '):
            strategies.append(col_str.replace("Strategies: ", "").strip())
        elif col_str == 'Primarily applies to the government':
            sectors.append(col_str.split(" ", 5)[-1].strip())
        elif col_str == 'Primarily applies to the private sector':
            sectors.append(col_str.split(" ", 6)[-2].strip())
    return {
        'applications': applications if applications else None,
        'harms': harms if harms else None,
        'incentives': incentives if incentives else None,
        'risks': risks if risks else None,
        'strategies': strategies if strategies else None,
        'sectors': sectors if sectors else None
    }

base = pathlib.Path(__file__).resolve().parent
asset_path = base / "assets"
corpus_path = asset_path / "corpus"
fulltext_path = corpus_path / "fulltext"
documents_path = corpus_path / "documents.csv"
segments_path = corpus_path / "segments.csv"
collections_path = corpus_path / "collections.csv"
authorities_path = corpus_path / "authorities.csv"

def merge_document_authorities(doc_col='Authority', auth_col='Name'):
    doc_df = pd.read_csv(documents_path)
    auth_df = pd.read_csv(authorities_path)
    df = pd.merge(doc_df, auth_df, left_on=doc_col, right_on=auth_col)
    return df

def merge_segment_document_authorities():
    doc_auth_df = merge_document_authorities()
    segment_df = pd.read_csv(segments_path)
    merged_seg_doc_auth = pd.merge(segment_df, doc_auth_df, left_on='Document ID', right_on='AGORA ID')
    return merged_seg_doc_auth

def get_raw_document_text(doc_id, path=fulltext_path):
    file_path = os.path.join(path, f'{doc_id}.txt')
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_all_document_texts(path=fulltext_path):
    texts = []
    if not os.path.exists(path):
        print(f'Warning: {path} does not exist')
        return texts
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read().strip())
            except Exception as e:
                print(e)
    return texts

if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

    long_docs = stream_long_summary_docs(text_splitter)
    print(f'Long summary documents')
    for i, doc in enumerate(long_docs):
        print(f'Document {i}: \n{doc}\n')

    short_docs = stream_short_summary_docs()
    print(f'Short summary documents')
    for i, doc in enumerate(short_docs):
        print(f'Document {i}: \n{doc}\n')

    segment_docs = stream_segment_docs(text_splitter)
    print(f'Segment documents')
    for i, doc in enumerate(segment_docs):
        print(f'Document {i}: \n{doc}\n')

    segment_summary_docs = stream_segment_summary_docs(text_splitter)
    print(f'Segment summary documents')
    for i, doc in enumerate(segment_summary_docs):
        print(f'Document {i}: \n{doc}\n')