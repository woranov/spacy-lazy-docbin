import spacy

from spacy_lazy_docbin import DocBin


def as_texts(docs):
    return [doc.text for doc in docs]


def test_docbin_indexing(nlp, docs):
    docbin = DocBin(docs=docs, vocab=nlp.vocab)

    assert len(docbin) == 10
    assert docbin[0].text == docs[0].text
    assert docbin[-1].text == docs[-1].text
    assert as_texts(docbin[2:5]) == as_texts(docs[2:5])
    assert as_texts(docbin[::-1]) == as_texts([doc for doc in docs[::-1]])
