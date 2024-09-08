import pytest
import spacy.tokens


@pytest.fixture
def nlp():
    en = spacy.util.get_lang_class("en")
    nlp = en(en().vocab)
    nlp.add_pipe("sentencizer")
    return nlp


@pytest.fixture
def docs(nlp):
    return [
        nlp.make_doc(" ".join(f"Doc {i} sentence {j}." for j in range(3)))
        for i in range(10)
    ]
