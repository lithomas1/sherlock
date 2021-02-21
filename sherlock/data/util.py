from typing import List

def tokenize_word(word: str) -> List[int]:
    """
    Maps each character in word to integer tokens.
    Numbers are mapped to themselves. Alphabet characters
    are mapped to 10-35
    Non alphanumeric characters are represented by an
    unknown token(id 36).

    :param word: str
        Input word to process
    :return: List[int]
        A List of character tokens

    """
    # Normalize
    word = word.lower()
    tokens = []
    for char in word:
        if char.isnumeric():
            tokens.append(int(char))
        elif char.isalpha() and char.isascii(): # Have to check for english :(
            idx = ord(char) - 97 + 10  # 97 is ASCII 'a'
            tokens.append(idx)
        else:
            tokens.append(36)

    return tokens