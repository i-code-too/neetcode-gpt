from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for num in numbers:
            number = str(num)
            tokens = []
            start = 0
            while start < len(number):
                for end in range(len(number), start, -1):
                    piece = number[start:end]
                    if piece in vocab:
                        tokens.append(piece)
                        start = end
                        break
            result.append(tokens)
        return result
            

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        start = 0
        while start < len(text):
            for end in range(len(text), start, -1):
                piece = text[start:end]
                if piece in vocab:
                    count += 1
                    start = end
                    break
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        token_count = self.count_tokens(text, vocab)
        word_count = len(text.split())
        fertility_score = token_count / word_count
        return round(fertility_score, 4)