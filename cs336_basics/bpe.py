import regex as re
import multiprocessing
from collections import defaultdict
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")

TEST_PAT = re.compile(r"\p{L}+\p{M}*|\p{N}+")

NUM_PROCESSES = 4
EOT = b"<|endoftext|>"

Pretoken = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]


def str_to_pretoken(s: str) -> Pretoken:
    return tuple(map(lambda i: i.to_bytes(), s.encode()))


def init_vocab():
    vocab = {i: i.to_bytes() for i in range(0, 256)}
    return vocab, len(vocab)


def pretokenize_chunk(
    input_path: str, start: int, end: int, special_token_pat: re.Pattern[str]
) -> tuple[
    dict[Pretoken, int],
    dict[BytePair, int],
    dict[BytePair, set[Pretoken]],
]:
    pretoken_counts: dict[Pretoken, int] = defaultdict(int)
    byte_pair_counts: dict[BytePair, int] = defaultdict(int)
    byte_pair_to_pretoken_index: dict[BytePair, set[Pretoken]] = defaultdict(set)

    with open(input_path, "rb") as f:
        _ = f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # make sure we do not chunk across special tokens
        for text in re.splititer(special_token_pat, chunk):
            for pretoken_match in re.finditer(PAT, text):
                pretoken = str_to_pretoken(pretoken_match.group())
                pretoken_counts[pretoken] += 1
                for i in range(len(pretoken) - 1):
                    byte_pair = (pretoken[i], pretoken[i + 1])
                    byte_pair_counts[byte_pair] += 1
                    byte_pair_to_pretoken_index[byte_pair].add(pretoken)

    return pretoken_counts, byte_pair_counts, byte_pair_to_pretoken_index


def pretokenize(
    input_path: str, special_tokens: list[str]
) -> tuple[
    dict[Pretoken, int],
    dict[BytePair, int],
    dict[BytePair, set[Pretoken]],
]:
    boundaries = find_chunk_boundaries(open(input_path, "rb"), NUM_PROCESSES, EOT)

    special_tokens_pat = re.compile("|".join(map(re.escape, special_tokens)))

    with multiprocessing.Pool(NUM_PROCESSES) as p:
        chunks = p.starmap(
            pretokenize_chunk, [(input_path, s, e, special_tokens_pat) for s, e in zip(boundaries, boundaries[1:])]
        )
        partial_pretoken_counts: list[dict[Pretoken, int]] = []
        partial_byte_pair_counts: list[dict[BytePair, int]] = []
        partial_byte_pair_to_pretoken_indices: list[dict[BytePair, set[Pretoken]]] = []
        for ppc, bpc, bppi in chunks:
            partial_pretoken_counts.append(ppc)
            partial_byte_pair_counts.append(bpc)
            partial_byte_pair_to_pretoken_indices.append(bppi)

    pretoken_counts: dict[Pretoken, int] = defaultdict(int)
    byte_pair_counts: dict[BytePair, int] = defaultdict(int)
    byte_pair_to_pretoken_index: dict[BytePair, set[Pretoken]] = defaultdict(set)

    for partial_pretoken_count in partial_pretoken_counts:
        for pretoken, count in partial_pretoken_count.items():
            pretoken_counts[pretoken] += count

    for partial_byte_pair_count in partial_byte_pair_counts:
        for byte_pair, count in partial_byte_pair_count.items():
            byte_pair_counts[byte_pair] += count

    for partial_byte_pair_to_pretoken_index in partial_byte_pair_to_pretoken_indices:
        for byte_pair, pretoken_index in partial_byte_pair_to_pretoken_index.items():
            byte_pair_to_pretoken_index[byte_pair].update(pretoken_index)

    return pretoken_counts, byte_pair_counts, byte_pair_to_pretoken_index


def merge_pretoken(byte_pair: BytePair, merge: bytes, pretoken: Pretoken) -> Pretoken:
    merged_pretoken_bytes: list[bytes] = []
    i = 0
    while i < len(pretoken):
        j = i + 1
        if j < len(pretoken) and (pretoken[i], pretoken[j]) == byte_pair:
            merged_pretoken_bytes.append(merge)
            i += 2
        else:
            merged_pretoken_bytes.append(pretoken[i])
            i += 1
    return tuple(merged_pretoken_bytes)


def overwrite_pretoken_counts(
    max_pair: BytePair,
    merge: bytes,
    pretoken_counts: dict[Pretoken, int],
    byte_pair_counts: dict[BytePair, int],
    byte_pair_to_pretoken_index: dict[BytePair, set[Pretoken]],
) -> None:
    affected_pretokens = list(byte_pair_to_pretoken_index[max_pair])

    for affected_pretoken in affected_pretokens:
        count = pretoken_counts[affected_pretoken]
        merged_pretoken = merge_pretoken(max_pair, merge, affected_pretoken)

        pairs: dict[BytePair, int] = defaultdict(int)

        for i in range(len(affected_pretoken) - 1):
            pair = affected_pretoken[i], affected_pretoken[i + 1]
            pairs[pair] += 1

        for pair, pair_count in pairs.items():
            byte_pair_counts[pair] -= pair_count * count

            assert affected_pretoken in byte_pair_to_pretoken_index[pair], (
                pair,
                affected_pretoken,
                byte_pair_counts[pair],
                byte_pair_to_pretoken_index[pair],
            )
            byte_pair_to_pretoken_index[pair].remove(affected_pretoken)

        pretoken_counts[merged_pretoken] += count
        del pretoken_counts[affected_pretoken]

        for i in range(len(merged_pretoken) - 1):
            pair = (merged_pretoken[i], merged_pretoken[i + 1])
            byte_pair_counts[pair] += count
            byte_pair_to_pretoken_index[pair].add(merged_pretoken)

    del byte_pair_counts[max_pair]
    del byte_pair_to_pretoken_index[max_pair]


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[BytePair]]:
    vocab, next_token_id = init_vocab()

    for special_token_string in special_tokens:
        vocab[next_token_id] = special_token_string.encode()
        next_token_id += 1

    merges: list[BytePair] = []

    pretoken_counts, byte_pair_counts, byte_pair_to_pretoken_index = pretokenize(input_path, special_tokens)

    rounds = vocab_size - len(vocab)

    for _ in tqdm(range(rounds)):
        pair_counts = [(v, k) for k, v in byte_pair_counts.items()]

        if not pair_counts:
            break

        _count, pair = max(pair_counts)

        merge = pair[0] + pair[1]
        vocab[next_token_id] = merge
        merges.append(pair)
        next_token_id += 1

        overwrite_pretoken_counts(pair, merge, pretoken_counts, byte_pair_counts, byte_pair_to_pretoken_index)

    return vocab, merges
