import re
from tokenizer import Tokenizer

with open('data.txt') as data:
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
    vocab = { token: idx for idx, token in enumerate(allWords) }
    # print(list(vocab.items())[10:100])

    t = Tokenizer(vocab)
    # Create ids vector
    ids = t.encode("Nagle wśród ciszy rozległ się huk jakby łoskot piorunu ponad głową kapitana")
    print(ids)

    # Decode ids vector
    text = t.decode(ids)
    print(text)

