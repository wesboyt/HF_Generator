import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast


def generate(model, tokenizer, original_input, num_sequences, max_length):
    input_ids = tokenizer(original_input, return_tensors="pt").input_ids.to('cuda')
    input_ids = input_ids.repeat(num_sequences, 1)
    length = input_ids.shape[1]
    for i in range(length, max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        likelihoods = torch.softmax(logits, 1)
        likelihoods[likelihoods < .02] = 0
        likelihoods /= likelihoods.sum(dim=1).unsqueeze(1)
        likelihoods = torch.cumsum(likelihoods, dim=1)
        roll = torch.rand(num_sequences).to('cuda').unsqueeze(1)
        likelihoods[likelihoods < roll] = 2
        tokens = likelihoods.argmin(dim=1)
        input_ids = torch.cat((input_ids, tokens.unsqueeze(1)), 1)
    return input_ids
