from cs336_basics.bpe import train_bpe

vocab_size = 32000
special_token = "<|endoftext|>"
file_path = "data/owt_train.txt"

if __name__ == "__main__":
    _ = train_bpe(file_path, vocab_size, [special_token])


