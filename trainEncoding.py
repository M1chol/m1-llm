import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

if len(sys.argv) < 2:
    print("example usecase:\n\ttrainEncoding.py data.txt data2.txt")
    quit()

print(f"Training on:\n\t{'\n\t'.join([item for item in sys.argv[1:]])}")

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
trainer = BpeTrainer(
    special_tokens=["<|unk|>", "<|eof|>"], # type: ignore
    show_progress = True # type: ignore
    )
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True) # type: ignore
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True) # type: ignore


print("Starting training...")
tokenizer.train(sys.argv[1:], trainer)
print("Training complete")

print("Vocab size:", tokenizer.get_vocab_size())

tokenizer.save("custom_tokenizer.json")
