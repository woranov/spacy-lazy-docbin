from typing import TYPE_CHECKING
from spacy_lazy_docbin.indexer import GroupedInnerAccessorMixin
from spacy_lazy_docbin.utils import docbin_sentence_counts

if TYPE_CHECKING:
    from spacy_lazy_docbin.base import DocBin, Sentence
    from spacy_lazy_docbin.chunked import DocBinChunks

class DocBinSentenceView(GroupedInnerAccessorMixin["Sentence"]):
    def __init__(self, docbin: "DocBin"):
        self.docbin = docbin

    @property
    def data(self):
        return [list(doc.sents) for doc in self.docbin]

    def counts(self) -> list[int]:
        return docbin_sentence_counts(self.docbin)


class DocBinChunksSentenceView(GroupedInnerAccessorMixin["Sentence"]):
    def __init__(self, docbin: "DocBinChunks"):
        self.docbin = docbin

    @property
    def data(self):
        return [DocBinSentenceView(docbin) for docbin in self.docbin.docbins]

    def counts(self) -> list[int]:
        return [
            sum(chunk_meta["sentences"]) for chunk_meta in self.docbin.meta["chunks"]
        ]


class SentenceView(GroupedInnerAccessorMixin["Sentence"]):
    def __init__(self, docbin: "DocBin | DocBinChunks"):
        from spacy_lazy_docbin.base import DocBin
        from spacy_lazy_docbin.chunked import DocBinChunks

        self.docbin = docbin
        if isinstance(docbin, DocBin):
            self._impl = DocBinSentenceView(docbin)
        elif isinstance(docbin, DocBinChunks):
            self._impl = DocBinChunksSentenceView(docbin)
        else:
            raise TypeError(
                f"Must be {DocBin.__qualname__} or {DocBinChunks.__qualname__}, not {type(docbin)}"
            )

    @property
    def data(self):
        return self._impl.data

    def counts(self) -> list[int]:
        return self._impl.counts()
