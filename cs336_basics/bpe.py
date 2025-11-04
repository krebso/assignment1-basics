import regex
import multiprocessing
from collections import defaultdict
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")

TEST_PAT = regex.compile(r"\p{L}+\p{M}*|\p{N}+")

NUM_PROCESSES = 4
EOT = b"<|endoftext|>"

Pretoken = tuple[bytes, ...]


def str_to_pretoken(s: str) -> Pretoken:
    return tuple(map(lambda i: i.to_bytes(), s.encode()))


def init_vocab():
    vocab = {i: i.to_bytes() for i in range(0, 256)}
    return vocab, len(vocab)


def pretokenize_chunk(
    input_path: str, start: int, end: int, special_token_pat: regex.Pattern[str]
) -> dict[Pretoken, int]:
    pretoken_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    
    with open(input_path, "rb") as f:
        _ = f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # make sure we do not chunk across special tokens
        for text in regex.splititer(special_token_pat, chunk):
            for pretoken_match in regex.finditer(PAT, text):
                pretoken = str_to_pretoken(pretoken_match.group())
                pretoken_counts[pretoken] += 1
    return pretoken_counts


def pretokenize(input_path: str, special_tokens: list[str]) -> dict[Pretoken, int]:
    boundaries = find_chunk_boundaries(open(input_path, "rb"), NUM_PROCESSES, EOT)

    special_tokens_pat = regex.compile("|".join(map(regex.escape, special_tokens)))

    with multiprocessing.Pool(NUM_PROCESSES) as p:
        partial_pretoken_counts = p.starmap(pretokenize_chunk, [(input_path, s, e, special_tokens_pat) for s, e in zip(boundaries, boundaries[1:])])

    pretoken_counts: dict[Pretoken, int] = defaultdict(int)

    for partial_pretoken_count in partial_pretoken_counts:
        for pretoken, count in partial_pretoken_count.items():
            pretoken_counts[pretoken] += count

    return pretoken_counts


def max_pair(pretoken_counts: dict[Pretoken, int]) -> tuple[bytes, bytes]:
    counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    for pretoken, count in pretoken_counts.items():
        for f, s in zip(pretoken, pretoken[1:]):  # TODO make more efficient
            counts[(f, s)] += count

    _, pair = max([(v, k) for k, v in counts.items()])

    return pair


def overwrite_pretoken_counts(pretoken_counts: dict[Pretoken, int], pair: tuple[bytes, bytes]) -> None:
    overwrites: list[tuple[tuple[bytes, ...], tuple[bytes, ...], int]] = []
    for pretoken, count in pretoken_counts.items():
        skip_next = False
        new_pretoken_bytes: list[bytes] = []
        i = 0
        while i < len(pretoken):
            if skip_next:
                skip_next = False
            elif i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                new_pretoken_bytes.append(pair[0] + pair[1])
                skip_next = True
            else:
                new_pretoken_bytes.append(pretoken[i])
            i += 1
        new_pretoken = tuple(new_pretoken_bytes)
        if new_pretoken != pretoken:
            overwrites.append((pretoken, new_pretoken, count))

    for old, new, count in overwrites:
        del pretoken_counts[old]
        pretoken_counts[new] = count


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab, next_token_id = init_vocab()

    for special_token_string in special_tokens:
        vocab[next_token_id] = special_token_string.encode()
        next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    pretoken_counts = pretokenize(input_path, special_tokens)

    rounds = vocab_size - len(vocab)

    for _ in tqdm(range(rounds)):
        pair = max_pair(pretoken_counts)

        merge = pair[0] + pair[1]
        vocab[next_token_id] = merge
        merges.append(pair)
        next_token_id += 1

        overwrite_pretoken_counts(pretoken_counts, pair)

    return vocab, merges
