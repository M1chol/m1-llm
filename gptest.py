import tiktoken

with open("data.txt") as data:
    raw = data.read()

tokenizer = tiktoken.get_encoding("gpt2")

encodedText = tokenizer.encode(raw) 

sample = encodedText[100:150]
# print(sample)
contextSize = 4
for i in range(1, contextSize+1):
    context = sample[:i]
    desired = sample[:i+1]
    print(tokenizer.decode(context), "->", tokenizer.decode(desired))
