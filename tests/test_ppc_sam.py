from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ppc.pipeline import _run_epoch, _sam_perturb, _sam_restore


class _DictLoader(TensorDataset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x, "label": y}


def _loader(n: int = 16, features: int = 6, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, features, generator=g)
    y = (x[:, 0] > 0).long()
    return DataLoader(_DictLoader(x, y), batch_size=n, shuffle=False)


@pytest.mark.parametrize("adaptive", [False, True])
def test_sam_perturb_has_radius_rho_and_restores_exactly(adaptive: bool) -> None:
    torch.manual_seed(0)
    model = nn.Linear(6, 2)
    batch = next(iter(_loader()))
    nn.CrossEntropyLoss()(model(batch["image"]), batch["label"]).backward()
    before = [p.detach().clone() for p in model.parameters()]

    offsets = _sam_perturb(list(model.parameters()), rho=0.05, adaptive=adaptive)
    assert offsets
    if adaptive:
        # ASAM constrains ||e / |w| || = rho
        norm = torch.sqrt(sum(((e / p0.abs().clamp_min(1e-12)) ** 2).sum() for (_, e), p0 in zip(offsets, before)))
    else:
        norm = torch.sqrt(sum((e ** 2).sum() for _, e in offsets))
    assert norm.item() == pytest.approx(0.05, rel=1e-4)

    _sam_restore(offsets)
    for p, p0 in zip(model.parameters(), before):
        torch.testing.assert_close(p.detach(), p0, rtol=0, atol=1e-7)


def test_sam_perturb_skips_non_finite_gradients() -> None:
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p.grad = torch.full_like(p, float("inf"))
    assert _sam_perturb(list(model.parameters()), rho=0.05, adaptive=False) == []


def test_sam_step_matches_reference_update() -> None:
    torch.manual_seed(1)
    model = nn.Linear(6, 2)
    reference = copy.deepcopy(model)
    loader = _loader()
    criterion = nn.CrossEntropyLoss()
    lr, rho = 0.1, 0.05

    _run_epoch(model, loader, criterion, torch.device("cpu"),
               torch.optim.SGD(model.parameters(), lr=lr), sam_rho=rho)

    # Reference: g1 at w, w_adv = w + rho*g1/||g1||, g2 at w_adv, w <- w - lr*g2
    batch = next(iter(loader))
    params = list(reference.parameters())
    g1 = torch.autograd.grad(criterion(reference(batch["image"]), batch["label"]), params)
    g1_norm = torch.sqrt(sum((g ** 2).sum() for g in g1))
    with torch.no_grad():
        for p, g in zip(params, g1):
            p.add_(rho * g / g1_norm)
    g2 = torch.autograd.grad(criterion(reference(batch["image"]), batch["label"]), params)
    with torch.no_grad():
        for p, g, gs in zip(params, g2, g1):
            p.sub_(rho * gs / g1_norm)
            p.sub_(lr * g)

    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-5, atol=1e-6)


def test_sam_updates_batchnorm_statistics_once_per_step() -> None:
    torch.manual_seed(2)
    base = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))
    plain, sam = copy.deepcopy(base), copy.deepcopy(base)
    loader = _loader()
    criterion = nn.CrossEntropyLoss()
    _run_epoch(plain, loader, criterion, torch.device("cpu"), torch.optim.SGD(plain.parameters(), lr=0.0))
    _run_epoch(sam, loader, criterion, torch.device("cpu"), torch.optim.SGD(sam.parameters(), lr=0.0), sam_rho=0.05)
    torch.testing.assert_close(sam[1].running_mean, plain[1].running_mean)
    torch.testing.assert_close(sam[1].running_var, plain[1].running_var)
    assert int(sam[1].num_batches_tracked) == int(plain[1].num_batches_tracked) + 1  # counter only
    assert sam[1].momentum == plain[1].momentum
