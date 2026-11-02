import torch

from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.model import NSModel
from nsrmp.train import loss_fn, train


def test_tiny_training_run_lowers_the_loss():
    batch = collate(make_dataset(256, seed=0))
    model = NSModel()
    before = _loss(model, batch)
    history = train(model, batch, epochs=8, seed=0)
    assert _loss(model, batch) < 0.85 * before
    assert len(history) == 8 and history[-1] < history[0]


def test_curriculum_trains_on_simple_instructions_first_and_can_be_switched_off():
    batch = collate(make_dataset(64, seed=1))
    seen = []
    train(NSModel(), batch, epochs=4, seed=0, curriculum=True, on_epoch=lambda e, size: seen.append(size))
    simple = (~batch.is_complex).sum().item()
    assert seen[0] == simple and seen[-1] == len(batch)
    seen.clear()
    train(NSModel(), batch, epochs=4, seed=0, curriculum=False, on_epoch=lambda e, size: seen.append(size))
    assert set(seen) == {len(batch)}


def _loss(model, batch):
    model.eval()
    with torch.no_grad():
        return loss_fn(model(batch)[0], batch.final, batch.mask).item()
