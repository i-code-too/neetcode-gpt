from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        vocab = list(corpus)
        merged = []
        for _ in range(num_merges):
            pairs = {}
            for i in range(len(vocab) - 1):
                each_pair = (vocab[i], vocab[i+1])
                pairs[each_pair] = pairs.get(each_pair, 0) + 1
            if not pairs:
                break

            ordered_pairs = sorted(pair for pair, count in pairs.items() if count == max(pairs.values()))
            best_pair = ordered_pairs[0]
            best_count = pairs[best_pair]
            merged.append([best_pair[0], best_pair[1]])

            new_vocab = []
            i=0
            while i < len(vocab):
                if i < len(vocab) - 1 and vocab[i] == best_pair[0] and vocab[i+1] == best_pair[1]:
                    new_vocab.append(best_pair[0] + best_pair[1]) # merging 'a', 'n' to 'an'
                    i += 2
                else:
                    new_vocab.append(vocab[i])
                    i += 1
            vocab = new_vocab
        return merged
