import torch

def orth(columns, tolerance=1e-07):
    if columns.numel() == 0:
        return columns[:, :0]
    u, s, _ = torch.linalg.svd(columns.double(), full_matrices=False)
    keep = s > max(tolerance * float(s[0]), 1e-12)
    return u[:, keep].to(columns.dtype)

def moment_basis(rows, cap):
    if rows.shape[0] == 0 or rows.numel() == 0:
        return rows.new_empty((rows.shape[-1], 0))
    h = rows.double()
    gram = h @ h.T / len(h)
    eig, vec = torch.linalg.eigh((gram + gram.T) * 0.5)
    ids = torch.where(eig > max(float(eig[-1]) * 1e-09, 1e-18))[0].flip(0)[:cap]
    if len(ids) == 0:
        return rows.new_empty((rows.shape[1], 0))
    result = h.T @ (vec[:, ids] / (len(h) * eig[ids]).sqrt())
    return orth(result).to(rows.dtype)

def decision_basis(rows, q, cap, rank, metric_fn, eta, epsilon, exclude=True, identity=False):
    projected = rows - rows @ q @ q.T if exclude and q.shape[1] else rows
    t = moment_basis(projected, cap)
    if not t.shape[1]:
        return (t, {'candidate_rank': 0, 'decision_rank': 0, 'lambda_d': 0.0})
    gc = (rows.double() @ t.double()).T @ (rows.double() @ t.double()) / len(rows)
    cc = torch.eye(t.shape[1], dtype=torch.float64) if identity else metric_fn(t).double().cpu()
    cc = (cc + cc.T) * 0.5
    damping = eta * (float(torch.trace(cc)) / len(cc) + epsilon)
    mc = cc + damping * torch.eye(len(cc), dtype=cc.dtype)
    e, u = torch.linalg.eigh(mc)
    invsqrt = u / e.clamp_min(epsilon).sqrt() @ u.T
    s = invsqrt @ gc @ invsqrt
    eig, vec = torch.linalg.eigh((s + s.T) * 0.5)
    ids = torch.where(eig > max(float(eig[-1]) * 1e-09, 1e-18))[0].flip(0)[:rank]
    v = orth(t.double() @ invsqrt @ vec[:, ids]).float()
    return (v, {'candidate_rank': t.shape[1], 'decision_rank': v.shape[1], 'lambda_d': damping, 'eigenvalues': eig[ids].tolist(), 'drift_metric': cc.tolist(), 'selection': 'identity' if identity else 'STE surrogate'})
