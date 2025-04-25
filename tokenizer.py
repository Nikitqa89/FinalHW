def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in s if c in stoi]

def decode(indices: list[int], itos: dict) -> str:
    return ''.join([itos.get(i, '') for i in indices])