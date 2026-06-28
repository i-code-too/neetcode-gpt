import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for _ in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:] # crop earlier tokens
            logits = model(context) # forward pass
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1) # extract last position
            next_token = torch.multinomial(probs, 1, generator=generator) # sample next token
            generator.set_state(initial_state)
            context = torch.cat((context, next_token), dim=-1) # append
            result.append(int_to_char[next_token.item()]) # decode token ID to character
        return ''.join(result)
