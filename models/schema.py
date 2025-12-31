from dataclasses import dataclass
from typing import Dict


@dataclass
class UnifiedDocument:
    source_id: str
    source_type: str   # "doc" | "web"
    title: str
    content: str
    metadata: Dict


@dataclass
class AnswerSource:
    source_type: str   # "doc" | "web"
    reference: str
