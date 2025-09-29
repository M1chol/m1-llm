import torch
from dataset import createDataLoader

vocab_size = 30000
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("data/orwell-rok-1984.txt", "r", encoding="utf-8") as file:
    rawText = file.read() 

max_length = 4
dataloader = createDataLoader(
            rawText, tokenizerFile="custom_tokenizer.json",
            batchSize=8, maxLength=max_length, stride=max_length,
            shuffle=False
        )

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)

context_len = max_length
pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_len))
# print(pos_embeddings.shape)

input_embedings = token_embeddings + pos_embeddings
# print(input_embedings.shape)