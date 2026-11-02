import torch

from nsrmp.baseline import NeuralBaseline
from nsrmp.batch import collate
from nsrmp.data import make_dataset
from nsrmp.train import train


def test_baseline_returns_a_distribution_over_real_objects_for_each_step():
    b = collate(make_dataset(8, seed=0, n_steps=(1, 3)))
    boxes, subj_w = NeuralBaseline()(b)
    assert boxes.shape == b.boxes.shape
    for k in range(8):
        n_steps = b.n_steps[k].item()
        active = subj_w[k, :n_steps]
        assert torch.allclose(active.sum(-1), torch.ones(n_steps), atol=1e-4)
        assert active[:, b.mask[k] == 0].abs().sum().item() == 0.0
        assert subj_w[k, n_steps:].abs().sum().item() == 0.0


def test_baseline_reads_the_words_not_the_parsed_program():
    b = collate(make_dataset(8, seed=1))
    model = NeuralBaseline().eval()
    scrambled = collate(make_dataset(8, seed=1))
    scrambled.programs = scrambled.programs[::-1]
    assert torch.allclose(model(b)[0], model(scrambled)[0])
    other_words = b.select(range(8))
    other_words.tokens = b.tokens.flip(0)
    other_words.lengths = b.lengths.flip(0)
    assert not torch.allclose(model(b)[0], model(other_words)[0])


def test_baseline_training_lowers_the_loss():
    # Note: this baseline does not learn to ground subjects from the final-box loss alone
    # (see README, "Baseline plateau"), so we only check that optimisation makes progress.
    data = collate(make_dataset(256, seed=2))
    model = NeuralBaseline()
    history = train(model, data, epochs=6, seed=0)
    assert history[-1] < 0.7 * history[0]
