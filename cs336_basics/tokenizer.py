from collections.abc import Iterable, Iterator
import json
from typing import Self, cast, final
import regex as re
from cs336_basics.bpe import PAT, Pretoken, str_to_pretoken


@final
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.int_to_bytes = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=lambda k: -len(k)) if special_tokens else None
        self.special_token_pat = (
            f"({'|'.join(re.escape(s) for s in self.special_tokens)})" if self.special_tokens is not None else None
        )
        self.bytes_to_int = {v: k for k, v in vocab.items()}

        if special_tokens is None:
            return

        next_token_id = len(vocab)

        for special_token_str in special_tokens:
            special_token = special_token_str.encode()
            if special_token not in self.bytes_to_int:
                self.int_to_bytes[next_token_id] = special_token
                self.bytes_to_int[special_token] = next_token_id
                next_token_id += 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        return cls(
            vocab=cast(dict[int, bytes], json.load(open(vocab_filepath))),
            merges=cast(list[tuple[bytes, bytes]], json.load(open(merges_filepath))),
            special_tokens=special_tokens,
        )

    def _merge_pretoken(self, pretoken: Pretoken) -> Pretoken:
        pretoken_byte_list = list(pretoken)
        new_pretoken_byte_list: list[bytes] = []
        for merge in self.merges:
            if len(pretoken_byte_list) == 1:
                break

            i = 0
            while i < len(pretoken_byte_list):
                if i < len(pretoken_byte_list) - 1:
                    f, s = pretoken_byte_list[i], pretoken_byte_list[i + 1]
                    if (f, s) == merge:
                        new_pretoken_byte_list.append(f + s)
                        i += 2
                    else:
                        new_pretoken_byte_list.append(f)
                        i += 1
                else:
                    new_pretoken_byte_list.append(pretoken_byte_list[i])
                    i += 1
            pretoken_byte_list, new_pretoken_byte_list = new_pretoken_byte_list, pretoken_byte_list
            new_pretoken_byte_list.clear()

        return tuple(pretoken_byte_list)

    def _encode_chunk(self, chunk: str, ids: list[int]) -> None:
        for pretoken_match in re.finditer(PAT, chunk):
            initial_pretoken = str_to_pretoken(pretoken_match.group())
            merged_pretoken = self._merge_pretoken(initial_pretoken)
            for b in merged_pretoken:
                ids.append(self.bytes_to_int[b])

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        if self.special_token_pat is None:
            self._encode_chunk(text, ids)
        else:
            assert self.special_tokens is not None
            for chunk in re.splititer(self.special_token_pat, text):
                if chunk in self.special_tokens:
                    ids.append(self.bytes_to_int[chunk.encode()])
                else:
                    self._encode_chunk(chunk, ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return (b"".join(self.int_to_bytes[i] for i in ids)).decode("utf-8", errors="replace")
