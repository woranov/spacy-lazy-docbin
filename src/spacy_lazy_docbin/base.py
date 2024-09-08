import os
from typing import Iterator, Self, Sequence, TypeVar, overload

from spacy.attrs import ORTH
from spacy.tokens import Doc, Span, DocBin as BaseDocBin
from spacy.vocab import Vocab
from spacy.tokens._dict_proxies import SpanGroups
import srsly

from spacy_lazy_docbin.indexer import _sanitize_index
from spacy_lazy_docbin.sentence_view import SentenceView
from spacy_lazy_docbin.utils import MISSING, Omittable


PathLike = os.PathLike | str
Sentence = TypeVar("Sentence", bound=Span)


class DocBin(BaseDocBin, Sequence[Doc]):
    def __init__(self, *args, vocab: Vocab, **kwargs):
        self.vocab = vocab
        super().__init__(*args, **kwargs)
        self._ensure_vocab_updated()
        self.sentences = SentenceView(self)

    def add(self, doc: Doc):
        super().add(doc)
        if doc.vocab is not self.vocab:
            self._ensure_vocab_updated()

    def merge(self, other: BaseDocBin):
        super().merge(other)
        self._ensure_vocab_updated()

    def from_bytes(self, bytes_data: bytes) -> Self:
        super().from_bytes(bytes_data)
        self._ensure_vocab_updated()
        return self

    def get_docs(self, vocab: Vocab | None = None) -> Iterator[Doc]:
        vocab = vocab or self.vocab
        self._update_vocab_if_foreign(vocab)

        for i in range(len(self)):
            yield self.get_doc(i, vocab)

    def get_doc(self, i: int, vocab: Vocab | None = None) -> Doc:
        vocab = vocab or self.vocab
        self._update_vocab_if_foreign(vocab)

        i = _sanitize_index(i, len(self))

        # from spacy.tokens.Doc.get_docs
        orth_col = self.attrs.index(ORTH)

        flags = self.flags[i]
        tokens = self.tokens[i]
        spaces = self.spaces[i]
        if flags.get("has_unknown_spaces"):
            spaces = None
        doc = Doc(vocab, words=tokens[:, orth_col], spaces=spaces)  # type: ignore
        doc = doc.from_array(self.attrs, tokens)  # type: ignore
        doc.cats = self.cats[i]
        # backwards-compatibility: may be b'' or serialized empty list
        if self.span_groups[i] and self.span_groups[i] != SpanGroups._EMPTY_BYTES:
            doc.spans.from_bytes(self.span_groups[i])
        else:
            doc.spans.clear()
        if i < len(self.user_data) and self.user_data[i] is not None:
            user_data = srsly.msgpack_loads(self.user_data[i], use_list=False)  # type: ignore
            doc.user_data.update(user_data)  # type: ignore

        return doc

    def _ensure_vocab_updated(
        self, strings: list | None = None, vocab: Vocab | None = None
    ):
        vocab = vocab or self.vocab

        for string in strings or self.strings:
            vocab[string]

    def _update_vocab_if_foreign(self, vocab: Vocab):
        if vocab is not self.vocab:
            self._ensure_vocab_updated(vocab=vocab)

    def _sanitize_index(self, i: int) -> int:
        orig_i = i
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(orig_i)

        return i

    @overload
    def __getitem__(self, k: int) -> Doc: ...
    @overload
    def __getitem__(self, k: slice) -> list[Doc]: ...
    def __getitem__(self, k: int | slice) -> Doc | list[Doc]:
        if isinstance(k, int):
            return self.get_doc(k, self.vocab)
        elif isinstance(k, slice):
            return list(self.islice(k))
        else:
            TypeError(k)

    def islice(
        self,
        stop_or_start: int | slice,
        stop: Omittable[int | None] = MISSING,
        step: int = 1,
        /,
        *,
        vocab: Vocab | None = None,
    ) -> Iterator[Doc]:
        if stop is MISSING:
            if isinstance(stop_or_start, slice):
                slc = stop_or_start
            else:
                slc = slice(stop_or_start)
        else:
            slc = slice(stop_or_start, stop, step)

        start, stop, step = slc.indices(len(self))

        for i in range(start, stop, step):
            yield self.get_doc(i, vocab or self.vocab)
