from typing import Self, TypeVar

from spacy.attrs import SENT_START
from spacy.tokens import DocBin


def _sanitize_index(i: int, length: int) -> int:
    orig_i = i
    if i < 0:
        i += length
    if i < 0 or i >= length:
        raise IndexError(orig_i)

    return i


def docbin_sentence_counts(docbin: DocBin) -> list[int]:
    return [
        (toks[:, docbin.attrs.index(SENT_START)] == 1).sum() for toks in docbin.tokens
    ]


class MissingType:
    def __new__(cls) -> Self:
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)

        return cls._instance

    def __repr__(self):
        return "<missing>"


T = TypeVar("T")
MISSING = MissingType()
Omittable = T | MissingType
