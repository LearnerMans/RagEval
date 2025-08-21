import dataclasses
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclasses.dataclass
class FileContext:
    file_name: str

@dataclasses.dataclass
class Chunk:
    text: str
    metadata: Optional[Dict[str, Any]] = None               

@dataclasses.dataclass
class ChunkedText:
    chunks: List[Chunk]

@dataclasses.dataclass
class ChunkerConfig:
    chunk_size: int
    chunk_overlap: int



class Chunker:
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    def chunk(self, text: str) -> ChunkedText:
        chunks = self.text_splitter.split_text(text)
        return ChunkedText(chunks=[Chunk(text=chunk) for chunk in chunks])

        



