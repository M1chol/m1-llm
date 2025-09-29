import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, maxLength, stride):
        self.inputIds = []
        self.targetIds = []

        tokenIds = tokenizer.encode(txt).ids

        for i in range(0, len(tokenIds) - maxLength, stride):
            inputChunk = tokenIds[i:i+maxLength]
            targetChunk = tokenIds[i+1:i+maxLength+1]
            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))

    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, idx):
        return self.inputIds[idx], self.targetIds[idx]


def createDataLoader(txt, tokenizerFile, batchSize=4, maxLength=256, stride=128, shuffle=True, dropLast=True, numWorkers=0):
    tokenizer = Tokenizer.from_file(tokenizerFile)
    dataset = GPTDataset(txt=txt, tokenizer=tokenizer, maxLength=maxLength, stride=stride)
    dataloader = DataLoader(
            dataset,
            batch_size = batchSize,
            shuffle = shuffle,
            drop_last = dropLast,
            num_workers = numWorkers
            )
    return dataloader
