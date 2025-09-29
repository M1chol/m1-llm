import tiktoken
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

raw = "Cześć, to jest test. Oto fragment przykładowego wejścia do tokenizera o kontekście szesnaście"

print("Full text:", raw)
print("Results using gpt2 tokenizer")

contextSize = 16
tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode(raw) 
print("Ids generated:", ids)

for i in range(1, contextSize+1):
    context = ids[:i]
    desired = ids[i]
    print(tokenizer.decode(context), "->", tokenizer.decode([desired]))

print("Results of using custom tokenizer")

# test if custom tokenizer gives better results
custom_tokenizer = Tokenizer.from_file("custom_tokenizer.json")
custom_tokenizer.decoder = ByteLevelDecoder()

ids = custom_tokenizer.encode(raw).ids
print("Ids generated:", ids)

for i in range(1, contextSize+1):
    context = ids[:i]
    desired = ids[i]
    print(custom_tokenizer.decode(context), "->", custom_tokenizer.decode([desired]))
