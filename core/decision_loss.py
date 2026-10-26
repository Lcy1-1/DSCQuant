import torch
import torch.nn.functional as F

def decision_loss(student, teacher, config):
    s, t = (student.float(), teacher.to(student.device).float())
    kl = F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction='sum')
    if config.kl_only or len(s) < 2:
        return kl
    top_values, top_indices = t.topk(2)
    if bool(top_values[0] == top_values[1]):
        return kl
    winner = int(top_indices[0])
    competitors = torch.arange(len(s), device=s.device)
    competitors = competitors[competitors != winner]
    rank = F.relu(torch.minimum(t.new_tensor(config.margin_cap), top_values[0] - top_values[1]) - s[winner] + s[competitors].max())
    count = min(config.siblings, len(competitors))
    chosen = competitors[t[competitors].topk(count).indices]
    teacher_gap = t[winner] - t[chosen]
    student_gap = s[winner] - s[chosen]
    sibling = (t[chosen].softmax(-1) * F.smooth_l1_loss(student_gap, teacher_gap, reduction='none', beta=1.0)).sum()
    return kl + config.rank_weight * rank + config.sibling_weight * sibling
