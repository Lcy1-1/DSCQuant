import math
import torch
from .model_ops import layers, block_output

def drift_metric(model, interface, candidate, records, pools, config, start_block):
    device = interface.rec_basis.device
    t = candidate.to(device)
    dimension = t.shape[1]
    result = torch.zeros(dimension, dimension, dtype=torch.float64)
    total_records = sum((len(states) for states in pools)) * len(layers(model))
    if total_records == 0:
        raise ValueError('Empty drift record pool.')
    previous = interface.override
    try:
        for context, (record, states) in enumerate(zip(records, pools)):
            for state in states:
                token_ids = list(record.prompt_ids) + list(state.prefix)
                for block_index in range(start_block, len(layers(model))):
                    denominator = math.sqrt(state.teacher_energy[block_index] + config.epsilon)

                    def surrogate(z):
                        interface.override = t @ z
                        return block_output(model, token_ids, block_index).float().flatten() / denominator
                    zero = t.new_zeros(dimension)
                    columns = []
                    for column in range(dimension):
                        direction = zero.clone()
                        direction[column] = 1.0
                        _, tangent = torch.autograd.functional.jvp(surrogate, zero, direction, create_graph=False, strict=False)
                        columns.append(tangent.detach().cpu())
                    j = torch.stack(columns, dim=1)
                    result += j.double().T @ j.double() / total_records
                    del j, columns
    finally:
        interface.override = previous
    return result
