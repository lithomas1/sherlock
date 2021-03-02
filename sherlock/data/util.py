from typing import List
import unicodedata


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
        elif char.isalpha() and char.isascii():  # Have to check for english :(
            idx = ord(char) - 97 + 10  # 97 is ASCII 'a'
            tokens.append(idx)
        else:
            tokens.append(36)

    return tokens


def sanitize_text(text: str) -> str:

    """
    Normalizes words and removes strange artifacts from the FEVER Dataset

    Currently removes accents and useless '-LRB-' and '-RRB' tags

    :param text: str or List[str]
        Text to sanitize
    :return: str
        Sanitized text.
    """
    if isinstance(text, list):
        for i, sent in enumerate(text):
            sent = sent.replace("-LRB-", "")
            sent = sent.replace("-LSB-", "")
            sent = sent.replace("-RRB-", "")

            # Accent removing
            # ref https://tinyurl.com/3kae7p6r (Stack Overflow)
            sent = unicodedata.normalize("NFKD", sent)
            text[i] = "".join([c for c in text if not unicodedata.combining(c)])
    else:
        text = text.replace("-LRB-", "")
        text = text.replace("-LRB-", "")
        text = text.replace("-RRB-", "")
        # Accent removing (see comment above)
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])

    return text