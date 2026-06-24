from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = sorted(set(text))
        stoi = {}
        itos = {}
        for i in range(len(chars)):
            stoi[chars[i]] = i
            itos[i] = chars[i]
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded = []
        for ch in text:
            encoded.append(stoi[ch])
        return encoded

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded = []
        for each_id in ids:
            decoded.append(itos[each_id])
        return ''.join(decoded)
