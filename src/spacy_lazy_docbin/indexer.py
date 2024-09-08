import abc
import bisect
import itertools
from typing import Iterator, Sequence, overload

from spacy_lazy_docbin.utils import MISSING, Omittable


def _sanitize_index(i: int, length: int) -> int:
    orig_i = i
    if i < 0:
        i += length
    if i < 0 or i >= length:
        raise IndexError(orig_i)

    return i


# TODO: less weird name
class GroupedInnerAccessorMixin[InnerT](abc.ABC):
    data: Sequence[Sequence[InnerT]]

    @abc.abstractmethod
    def counts(self) -> list[int]: ...

    @overload
    def __getitem__(self, k: int) -> InnerT: ...
    @overload
    def __getitem__(self, k: slice) -> list[InnerT]: ...
    def __getitem__(self, k: int | slice) -> InnerT | list[InnerT]:
        if isinstance(k, int):
            outer_idx, inner_idx = self._indices_for_inner_index(k)
            return self.data[outer_idx][inner_idx]
        elif isinstance(k, slice):
            return list(self.islice(k))
        else:
            TypeError(k)

    def __len__(self) -> int:
        return sum(self.counts())

    def _indices_for_inner_index(self, i: int) -> tuple[int, int]:
        inner_idx = _sanitize_index(i, len(self))
        accumulated = list(itertools.accumulate(self.counts()))
        outer_idx = bisect.bisect_right(accumulated, inner_idx)
        inner_idx = inner_idx - (accumulated[outer_idx - 1] if outer_idx > 0 else 0)

        return outer_idx, inner_idx

    def islice(
        self,
        stop_or_start: int | slice,
        stop: Omittable[int | None] = MISSING,
        step: int = 1,
        /,
    ) -> Iterator[InnerT]:
        if stop is MISSING:
            if isinstance(stop_or_start, slice):
                slc = stop_or_start
            else:
                slc = slice(stop_or_start)
        else:
            slc = slice(stop_or_start, stop, step)

        start, stop, step = slc.indices(len(self))

        if step == 1:
            outer_idx_start, inner_idx_start = self._indices_for_inner_index(start)
            inner_idx = stop - 1
            for i in range(start + 1, stop):
                outer_idx, inner_idx = self._indices_for_inner_index(i)
                if outer_idx != outer_idx_start:
                    # wrapped over to the next doc
                    # we can yield the sentences in bulk
                    yield from itertools.islice(
                        self.data[outer_idx_start], inner_idx_start, None
                    )

                    outer_idx_start, inner_idx_start = outer_idx, inner_idx

            yield from itertools.islice(
                self.data[outer_idx_start], inner_idx_start, inner_idx + 1
            )

        else:
            for i in range(start, stop, step):
                outer_idx, inner_idx = self._indices_for_inner_index(i)
                yield self.data[outer_idx][inner_idx]
