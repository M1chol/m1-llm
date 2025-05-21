import tiktoken
from tokenizers import Tokenizer


with open("data.txt") as data:
    raw = data.read()

print("Results using gpt2 tokenizer")

tokenizer = tiktoken.get_encoding("gpt2")

encodedText = tokenizer.encode(raw) 

sample = encodedText[100:150]
# print(sample)
contextSize = 4
for i in range(1, contextSize+1):
    context = sample[:i]
    desired = sample[:i+1]
    print(tokenizer.decode(context), "->", tokenizer.decode(desired))

print("Results of using custom tokenizer")

# test if custom tokenizer gives better results
tokenizer2 = Tokenizer.from_file("custom_tokenizer.json")

encodedText = tokenizer2.encode(raw)
sample = encodedText.ids[120:150]

for i in range(1, contextSize+1):
    context = sample[:i]
    desired = sample[:i+1]
    print(tokenizer2.decode(context), "->", tokenizer2.decode(desired))
