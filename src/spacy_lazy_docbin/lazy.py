from os import PathLike
from pathlib import Path

from spacy_lazy_docbin.base import BaseDocBin, DocBin
from spacy_lazy_docbin.logging import logger


class LazyDocBin(DocBin):
    def __init__(self, *args, path: PathLike, **kwargs):
        self.path = Path(path)
        self._loaded = False
        super().__init__(*args, **kwargs)

    def _load(self):
        if not self._loaded:
            if self.path.exists():
                # we allow for docs being added before loading the file
                if self.tokens:
                    existing_docs_bin = BaseDocBin(
                        attrs=self.attrs,
                        store_user_data=self.store_user_data,
                        docs=self.get_docs(),
                    )
                else:
                    existing_docs_bin = None

                logger.info(f"Loading DocBin from {self.path}")
                self.from_disk(self.path)
                if existing_docs_bin:
                    self.merge(existing_docs_bin)
            else:
                super().to_disk(self.path)

            self._loaded = True

    def save_and_unload(self):
        self.to_disk(self.path)
        self.tokens = []
        self.spaces = []
        self.cats = []
        self.span_groups = []
        self.user_data = []
        self.flags = []
        self.strings = set()
        self._loaded = False

    def to_disk(self, path: PathLike):
        self._load()
        super().to_disk(Path(path))
