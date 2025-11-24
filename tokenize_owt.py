from cs336_basics.bpe import train_bpe
import pickle

vocab_size = 32000
special_token = "<|endoftext|>"
file_path = "data/owt_train.txt"

if __name__ == "__main__":
    vocab, merges = train_bpe(file_path, vocab_size, [special_token])
    pickle.dump(vocab, open("owt_vocab.json", "wb"))
    pickle.dump(merges, open("owt_merges.json", "wb"))


