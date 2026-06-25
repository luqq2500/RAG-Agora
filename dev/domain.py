from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentMetadata:
    agora_id: str
    authority: str
    collection: Optional[str]
    status: Optional[str]
    applications: Optional[str]
    harms: Optional[str]
    incentives: Optional[str]
    risks: Optional[str]
    strategies: Optional[str]
    sectors: Optional[str]

@dataclass
class TextSplitMetadata:
    current_chunk: int
    total_chunks: int