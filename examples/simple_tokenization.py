import re

class Tokenizer:
    def __init__(self, vocab):
        self._stringToInt = vocab
        self._intToString = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\',.]|-|--|\s)', text)
        preproc = [i for i in preproc if i.strip()]
        preproc = [item if item in self._stringToInt else '<|unk|>' for item in preproc]
        ids = [self._stringToInt[token] for token in preproc]
        return ids

    def decode(self, tokenIds : list):
        text = " ".join([ self._intToString[tId] for tId in tokenIds ])
        return re.sub(r'\s+([,.!?"()\'])', r'\1', text)


with open('data/pan-tadeusz.txt') as data:
    raw = data.read()
    print(f'oryginalny tekst: {len(raw)} znaków')

    # original text split by words with diferencing symbols
    preproc = re.split(r'([,.:;?_!"()\',.]|-|--|\s)',raw)
    # print(preproc[10:100])

    # removing spaces from list
    preproc = [i for i in preproc if i.strip()]
    # print(preproc[10:100])
    print(f'wygenerowano {len(preproc)} tokenów')

    # Generating token id's
    allWords = sorted(set(preproc))
    allWords.extend(['<|eof|>','<|unk|>'])
    vocab = { token: idx for idx, token in enumerate(allWords) }
    # print(list(vocab.items())[10:100])

    t = Tokenizer(vocab)
    # Create ids vector
    ids = t.encode("Gdzieś w otchłani rozległ się huk jakby łoskot piorunu ponad głową kapitana")
    print(ids)

    # Decode ids vector
    text = t.decode(ids)
    print(text)

