"""
Data models for the vector store (Pinecone).
"""

from pydantic import BaseModel, Field

from src.models.enums import ChunkType


class ChunkMetadata(BaseModel):
    """Metadata attached to each vector in Pinecone. Supports filtering."""

    client_id: str
    source_file: str
    paragraph_index: int
    chunk_type: ChunkType
    has_spec_value: bool = False
    products_mentioned: list[str] = Field(default_factory=list)
    regions_mentioned: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A text chunk ready for embedding and storage in Pinecone."""

    id: str
    text: str
    metadata: ChunkMetadata
