from collections import defaultdict


def create_index(tokens):
    index = defaultdict(list)
    for token_index, token in enumerate(tokens):
        index[token].append(token_index)
    return index
