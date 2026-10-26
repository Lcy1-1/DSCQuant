import torch

def layers(model):
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise TypeError('Expected LlamaForCausalLM with model.layers; NCI is not supported.')
    return model.model.layers

class _Captured(Exception):

    def __init__(self, value):
        self.value = value

def model_inputs(model, token_ids):
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'use_cache': False}

def block_output(model, token_ids, index):

    def hook(module, args, output):
        raise _Captured(output[0] if isinstance(output, tuple) else output)
    handle = layers(model)[index].register_forward_hook(hook)
    try:
        model(**model_inputs(model, token_ids))
    except _Captured as captured:
        return captured.value
    finally:
        handle.remove()
    raise RuntimeError('Decoder block was not reached.')
