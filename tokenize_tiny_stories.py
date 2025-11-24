import pickle
from cs336_basics.bpe import train_bpe

vocab_size = 10000
special_token = "<|endoftext|>"
file_path = "data/TinyStoriesV2-GPT4-train.txt"

if __name__ == "__main__":
    vocab, merges = train_bpe(file_path, vocab_size, [special_token])
    pickle.dump(vocab, open("tiny_stories_vocab.json", "wb"))
    pickle.dump(merges, open("tiny_stories_merges.json", "wb"))


