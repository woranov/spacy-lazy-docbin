from pathlib import Path
from typing import cast, Self, Sequence, TypedDict

from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import SimpleFrozenList

import srsly

from spacy_lazy_docbin.base import PathLike
from spacy_lazy_docbin.indexer import GroupedInnerAccessorMixin
from spacy_lazy_docbin.lazy import LazyDocBin
from spacy_lazy_docbin.sentence_view import SentenceView
from spacy_lazy_docbin.utils import docbin_sentence_counts


class ChunkInfo(TypedDict):
    size: int
    offset: int
    sentences: list[int]
    path: str


class ChunksTotalsInfo(TypedDict):
    size: int
    sentences: list[int]


class ChunksInfo(TypedDict):
    chunk_size: int
    chunks: list[ChunkInfo]
    totals: ChunksTotalsInfo


class DocBinChunks(GroupedInnerAccessorMixin[Doc]):
    def __init__(
        self,
        vocab: Vocab,
        path: PathLike,
        meta: ChunksInfo,
        keep_last_n_chunks: int = -1,
        docbins: Sequence[LazyDocBin] = SimpleFrozenList(),
        num_padding=2,
        **docbin_init_kwargs,
    ):
        self.vocab = vocab
        self.path = Path(path)
        self.meta = meta
        self.keep_last_n_chunks = keep_last_n_chunks
        self.docbins = list(docbins)
        self.num_padding = num_padding
        self.docbin_init_kwargs = docbin_init_kwargs
        self.sentences = SentenceView(self)

    @property
    def data(self) -> Sequence[LazyDocBin]:
        return self.docbins

    @classmethod
    def load(
        cls,
        vocab: Vocab,
        path: PathLike,
        keep_last_n_chunks=-1,
        docbin_init_kwargs=None,
    ) -> Self:
        path = Path(path)
        meta = cast(ChunksInfo, srsly.read_json(path))

        docbins = [
            LazyDocBin(vocab=vocab, path=Path(chunk["path"]))
            for chunk in meta["chunks"]
        ]

        return cls(
            vocab=vocab,
            path=path,
            meta=meta,
            docbins=docbins,
            keep_last_n_chunks=keep_last_n_chunks,
            **(docbin_init_kwargs or {}),
        )

    @classmethod
    def create(
        cls,
        vocab: Vocab,
        path: PathLike,
        chunk_size: int,
        keep_last_n_chunks=-1,
        **docbin_init_kwargs,
    ) -> Self:
        path = Path(path)
        meta: ChunksInfo = {
            "chunk_size": chunk_size,
            "chunks": [],
            "totals": {"size": 0, "sentences": []},
        }

        return cls(
            vocab=vocab,
            path=path,
            meta=meta,
            keep_last_n_chunks=keep_last_n_chunks,
            **docbin_init_kwargs,
        )

    def counts(self) -> list[int]:
        return [len(docbin) for docbin in self.docbins]

    def add_chunk(self, docbin: LazyDocBin):
        size = len(docbin)
        offset = self.meta["totals"]["size"]
        sent_counts = docbin_sentence_counts(docbin)
        docbin_filename = f"chunk-{{:0{self.num_padding}d}}.spacy".format(
            len(self.docbins)
        )

        self.docbins.append(docbin)
        self.meta["chunks"].append(
            {
                "size": size,
                "offset": offset,
                "sentences": sent_counts,
                "path": str((self.path / docbin_filename).absolute()),
            }
        )
        self.meta["totals"]["size"] += size
        self.meta["totals"]["sentences"] += [sum(sent_counts)]

        if self.keep_last_n_chunks > 0:
            for docbin in self.docbins[: -self.keep_last_n_chunks]:
                docbin.save_and_unload()

    def add(self, doc: Doc):
        if not self.docbins:
            path = self.path / "chunk-{}.spacy".format("0" * self.num_padding)
            docbin = LazyDocBin(**self.docbin_init_kwargs, vocab=self.vocab, path=path)
            docbin.add(doc)
            self.add_chunk(docbin)
        else:
            last_chunk = self.docbins[-1]
            if len(last_chunk) >= self.meta["chunk_size"]:
                self.add_chunk(LazyDocBin(**self.docbin_init_kwargs, path=self.path))

            last_chunk.add(doc)

        sent_count = len(list(doc.sents))
        self.meta["chunks"][-1]["size"] += 1
        self.meta["chunks"][-1]["sentences"] += [sent_count]
        self.meta["totals"]["size"] += 1
        self.meta["totals"]["sentences"][-1] += sent_count
