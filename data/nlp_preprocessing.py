import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        full_dataset = positive + negative
        vocabulary = set()
        for sentence in full_dataset:
            vocabulary.update(sentence.split())
        vocab_sorted = sorted(vocabulary)

        token_id = {} # hash map
        for i in range(len(vocab_sorted)):
            token_id.update({vocab_sorted[i]: i+1})
        
        encoded_tensors = []
        for sentence in full_dataset:
            encoded = [token_id[word] for word in sentence.split()]
            encoded_tensors.append(torch.tensor(encoded))

        return nn.utils.rnn.pad_sequence(encoded_tensors, padding_value = 0, batch_first = True)
