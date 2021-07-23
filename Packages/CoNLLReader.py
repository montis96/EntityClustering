import re
from pathlib import Path

"""
Parsing del file CONLL
"""


def read_aida_yago_conll(path):
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf-8')





    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, _, tag = line.split('\t')
            tokens.append(token.lower())
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs
