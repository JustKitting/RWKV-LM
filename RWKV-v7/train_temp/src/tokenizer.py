import ast


class RWKVTokenizer:
    def __init__(self, vocab_file: str):
        self.idx2token = {}
        sorted_tokens = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                space = line.index(" ")
                idx = int(line[:space])
                last_space = line.rindex(" ")
                token_bytes = ast.literal_eval(line[space:last_space])
                if isinstance(token_bytes, str):
                    token_bytes = token_bytes.encode("utf-8")
                sorted_tokens.append(token_bytes)
                self.idx2token[idx] = token_bytes

        self.token2idx = {tok: idx for idx, tok in self.idx2token.items()}
        self.vocab_size = len(self.idx2token)

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for token in reversed(sorted_tokens):
            if len(token) < 2:
                continue
            first = token[0]
            second = token[1]
            self.table[first][second].append(token)
            self.wlen[first] = max(self.wlen[first], len(token))
            self.good[first].add(second)

    def encode(self, text: str):
        src = text.encode("utf-8")
        tokens = []
        i = 0
        src_len = len(src)
        while i < src_len:
            candidate = src[i : i + 1]
            if i < src_len - 1:
                first = src[i]
                second = src[i + 1]
                if second in self.good[first]:
                    window = src[i : i + self.wlen[first]]
                    for match in self.table[first][second]:
                        if window.startswith(match):
                            candidate = match
                            break
            tokens.append(self.token2idx[candidate])
            i += len(candidate)
        return tokens

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        byte_chunks = []
        append = byte_chunks.append
        for tok in tokens:
            piece = self.idx2token.get(int(tok))
            if piece is None:
                continue
            append(piece)
        return b"".join(byte_chunks).decode("utf-8", errors="ignore")
