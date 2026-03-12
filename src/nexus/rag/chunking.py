"""
Document chunking strategies for RAG.

Provides multiple chunking approaches:
- Fixed size: Simple character/token-based chunks
- Recursive: Smart splitting on natural boundaries
- Semantic: Split on topic/meaning changes
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Chunker(ABC):
    """Base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(Chunker):
    """
    Split text into fixed-size chunks with overlap.

    Simple but effective for many use cases.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: str = "characters",  # "characters" or "tokens"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def _get_length(self, text: str) -> int:
        if self.length_function == "tokens":
            # Rough approximation: 4 chars per token
            return len(text) // 4
        return len(text)

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Find end position
            end = start + self.chunk_size

            # Don't split mid-word if possible
            if end < len(text):
                # Look for word boundary
                space_pos = text.rfind(" ", start, end)
                if space_pos > start + self.chunk_size // 2:
                    end = space_pos + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text) or start <= chunks[-1].start_char if chunks else False:
                break

        return chunks


class RecursiveChunker(Chunker):
    """
    Recursively split text on natural boundaries.

    Tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences
    4. Words

    This preserves semantic structure better than fixed-size chunking.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "? ",
        "! ",
        "; ",
        ", ",
        " ",     # Words
        "",      # Characters (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = self._split_text(text, self.separators)

        return [
            Chunk(
                text=chunk_text,
                start_char=0,  # Would need tracking for accurate positions
                end_char=len(chunk_text),
                chunk_index=i,
                metadata=metadata.copy() if metadata else {},
            )
            for i, chunk_text in enumerate(chunks)
            if chunk_text.strip()
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            splits = list(text)
        else:
            splits = text.split(separator)

        # Merge small chunks and split large ones
        chunks = []
        current_chunk = ""

        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk)

                # Handle oversized splits
                if len(split) > self.chunk_size:
                    # Recursively split with finer separators
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class SentenceChunker(Chunker):
    """
    Split text into sentence-based chunks.

    Groups sentences together until chunk_size is reached.
    Good for maintaining semantic coherence.
    """

    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # Number of sentences to overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_sentences.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            else:
                # Save current chunk
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_char=0,
                        end_char=len(chunk_text),
                        chunk_index=chunk_index,
                        metadata=metadata.copy() if metadata else {},
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_start = max(0, len(current_sentences) - self.chunk_overlap)
                current_sentences = current_sentences[overlap_start:] + [sentence]
                current_length = sum(len(s) + 1 for s in current_sentences)

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=0,
                end_char=len(chunk_text),
                chunk_index=chunk_index,
                metadata=metadata.copy() if metadata else {},
            ))

        return chunks


class MarkdownChunker(Chunker):
    """
    Chunk markdown documents by headers.

    Preserves document structure by keeping sections together.
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 1024,
        include_headers: bool = True,
    ):
        self.chunk_size = chunk_size
        self.include_headers = include_headers

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = []

        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(text))

        if not headers:
            # No headers, use recursive chunking
            fallback = RecursiveChunker(chunk_size=self.chunk_size)
            return fallback.chunk(text, metadata)

        # Split by headers
        sections = []
        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            header_level = len(match.group(1))
            header_text = match.group(2)
            content = text[start:end]

            sections.append({
                "level": header_level,
                "header": header_text,
                "content": content,
            })

        # Create chunks from sections
        chunk_index = 0
        for section in sections:
            content = section["content"]

            if len(content) <= self.chunk_size:
                chunks.append(Chunk(
                    text=content,
                    start_char=0,
                    end_char=len(content),
                    chunk_index=chunk_index,
                    metadata={
                        **(metadata or {}),
                        "header": section["header"],
                        "header_level": section["level"],
                    },
                ))
                chunk_index += 1
            else:
                # Section too large, sub-chunk it
                sub_chunker = RecursiveChunker(chunk_size=self.chunk_size)
                sub_chunks = sub_chunker.chunk(content)

                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.metadata = {
                        **(metadata or {}),
                        "header": section["header"],
                        "header_level": section["level"],
                    }
                    chunks.append(sub_chunk)
                    chunk_index += 1

        return chunks


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> Chunker:
    """
    Factory function to get a chunker.

    Args:
        strategy: One of "fixed", "recursive", "sentence", "markdown"
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        **kwargs: Additional arguments for the chunker

    Returns:
        Chunker instance
    """
    chunkers = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "markdown": MarkdownChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(chunkers.keys())}")

    return chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
