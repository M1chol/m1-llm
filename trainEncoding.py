import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

if len(sys.argv) < 2:
    print("example usecase:\n\ttrainEncoding.py data.txt data2.txt")
    quit()

print(f"Training on:\n\t{'\n\t'.join([item for item in sys.argv[1:]])}")

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
trainer = BpeTrainer(special_tokens=["<|unk|>", "<|eof|>"])
tokenizer.pre_tokenizer = BertPreTokenizer()

print("Starting training...")
tokenizer.train(sys.argv[1:], trainer)
print("Training complete")

tokenizer.save("custom_tokenizer.json")
