import torch


def flatten_if_batched(*tensors):
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    if len(first.shape) not in (2, 3):
        raise AssertionError("Input tensors must be 2D or 3D")

    def _contig(t):
        return t if t.is_contiguous() else t.contiguous()

    if len(first.shape) == 3:
        batched = True
        batch_size = first.shape[0]
        assert all(t.shape[0] == batch_size for t in tensors)
        assert all(t.shape[1] == first.shape[1] for t in tensors)
        flat_tensors = [_contig(t).reshape(-1, t.shape[-1]) for t in tensors]
    else:
        batched = False
        batch_size = None
        flat_tensors = [_contig(t) for t in tensors]

    return flat_tensors, batched, batch_size

