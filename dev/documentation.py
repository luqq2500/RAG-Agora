import os
import pandas as pd
from langchain_core.documents import Document
from pandas import Series


def merge_document_authorities(doc_path, auth_path, doc_col='Authority', auth_col='Name'):
    doc_df = pd.read_csv(doc_path)
    auth_df = pd.read_csv(auth_path)
    return pd.merge(doc_df, auth_df, left_on=doc_col, right_on=auth_col)

def get_raw_document_text(doc_id, path='assets/corpus/texts/'):
    file_path = os.path.join(path, f'{doc_id}.txt')
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def prepare_doc_document(doc_path, auth_path) -> list[Document]:
    df = merge_document_authorities(doc_path, auth_path)
    documents = []

    for index, row in df.iterrows():
        document_id = row.get("AGORA ID")
        if pd.isna(document_id):
            continue

        raw_document_text = get_raw_document_text(str(document_id))
        overview = prepare_overview_semantic(row)
        usecase_flags = collect_usecase_flags(row)
        usecases = prepare_usecase_semantic(usecase_flags)

        content = f'{overview} {usecases} {raw_document_text}'.strip()

        metadata = {
            'AGORA ID': row.get("AGORA ID"),
            'Official Name': row.get("Official Name"),
            'Casual Name': row.get("Casual Name"),
            'Authority': row.get("Authority"),
            'Collection': row.get("Collection"),
            'Proposed Date': row.get("Proposed date"),
            'Recent Activity': row.get("Recent activity"),
            **{category: usecases for category, usecases in usecase_flags.items()},
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def prepare_usecase_semantic(usecase_flags: dict[str, list[str]]):
    semantics = []
    if usecases := usecase_flags.get('Applications'):
        semantics.append(f'It specifically applies to the following domains: {", ".join(usecases)}.')
    if usecases := usecase_flags.get('Risks'):
        semantics.append(f'It identifies concerns and risk factors regarding: {", ".join(usecases)}.')
    if usecases := usecase_flags.get('Harms'):
        semantics.append(f'The operational goal is to mitigate harms on: {", ".join(usecases)}.')
    if usecases := usecase_flags.get('Strategies'):
        semantics.append(f'The regulatory strategy involves: {", ".join(usecases)}.')
    if usecases := usecase_flags.get('Incentives'):
        semantics.append(f'Enforcement relies on compliance mechanisms such as: {", ".join(usecases)}.')
    return " ".join(semantics)

def collect_usecase_flags(row: Series) -> dict[str, list[str]]:
    applications = []
    harms = []
    incentives = []
    risks = []
    strategies = []
    for col, value in row.items():
        if value is True or str(value).lower() == 'true':
            col_str = str(col)
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
    return {'Applications': applications, 'Harms': harms, 'Incentives': incentives, 'Risks': risks, 'Strategies': strategies}


def prepare_overview_semantic(row: Series) -> str:
    official_name = row.get('Official name', "Unnamed Official")
    casual_name = row.get('Casual name', "Unnamed Casual")
    source = row.get('Link to document', 'No source')
    authority = row.get('Authority', "Unnamed Authority")
    collection = row.get('Collection', "Unnamed Collection")
    proposed_on = row.get('Proposed date', 'Undisclosed Proposed Date')
    recent_activity = row.get('Most recent activity', 'Undisclosed Recent Activity')
    recent_activity_date = row.get('Most recent activity date', 'Undisclosed Recent Activity Date')
    overview = (
        f"This AI policy document is titled {casual_name} under official title of {official_name}, issued by {authority} under collection of {collection}, sourced from {source}. "
        f"It was proposed on {proposed_on}, with recent activity of {recent_activity} on {recent_activity_date}")
    return overview

