import re

class Tokenizer:
    def __init__(self, vocab):
        self._stringToInt = vocab
        self._intToString = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\',.]|-|--|\s)', text)
        preproc = [i for i in preproc if i.strip()]
        return [ self._stringToInt[token] for token in preproc if token ]
    
    def decode(self, tokenIds : list):
        text = " ".join([ self._intToString[tId] for tId in tokenIds ])
        return re.sub(r'\s+([,.!?"()\'])', r'\1', text)
