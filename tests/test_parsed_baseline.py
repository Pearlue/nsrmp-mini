import torch

from nsrmp.baseline import ParsedNeuralBaseline
from nsrmp.batch import ROLE_GROUPS, collate
from nsrmp.data import make_dataset
from nsrmp.metrics import evaluate
from nsrmp.train import train


def test_parsed_baseline_returns_a_distribution_over_real_objects_for_each_step():
    b = collate(make_dataset(8, seed=0, n_steps=(1, 3)))
    boxes, subj_w = ParsedNeuralBaseline()(b)
    assert boxes.shape == b.boxes.shape
    for k in range(8):
        n_steps = b.n_steps[k].item()
        active = subj_w[k, :n_steps]
        assert torch.allclose(active.sum(-1), torch.ones(n_steps), atol=1e-4)
        assert active[:, b.mask[k] == 0].abs().sum().item() == 0.0
        assert subj_w[k, n_steps:].abs().sum().item() == 0.0


def test_parsed_baseline_reads_the_role_words_not_the_sentence_or_the_program():
    b = collate(make_dataset(8, seed=1, p_complex=1.0))
    torch.manual_seed(0)
    model = ParsedNeuralBaseline().eval()
    other = b.select(range(8))
    other.programs = other.programs[::-1]
    other.tokens, other.lengths = b.tokens.flip(0), b.lengths.flip(0)
    assert torch.allclose(model(b)[0], model(other)[0])
    for g in range(len(ROLE_GROUPS)):
        swapped = b.select(range(8))
        swapped.role_tokens = b.role_tokens.clone()
        swapped.role_tokens[:, :, g] = b.role_tokens[:, :, g].flip(0)
        # at initialisation some groups change the output only slightly, so test for any change
        assert not torch.equal(model(b)[0], model(swapped)[0]), ROLE_GROUPS[g]


def test_parsed_baseline_training_lowers_the_loss():
    data = collate(make_dataset(256, seed=2))
    model = ParsedNeuralBaseline()
    history = train(model, data, epochs=12, seed=0)
    assert history[-1] < 0.8 * history[0]


def test_parsed_baseline_attention_depends_only_on_the_words_when_there_is_no_relation():
    # Nothing word-independent may rank objects, otherwise attention collapses onto an arbitrary
    # object before grounding is learned (see README, "Why the attention design matters").
    b = collate(make_dataset(8, seed=4, n_objects=(5, 5), p_complex=0.0))
    b.role_tokens[:, :, 0] = 0  # empty subject head, and simple phrases have no relation
    torch.manual_seed(0)
    _, subj_w = ParsedNeuralBaseline()(b)
    assert torch.allclose(subj_w[:, 0], torch.full((8, 5), 0.2), atol=1e-6)


def test_parsed_baseline_learns_to_ground_subjects_and_move_them_from_final_boxes_alone():
    data = collate(make_dataset(1000, seed=5))
    test = collate(make_dataset(200, seed=6))
    torch.manual_seed(0)
    model = ParsedNeuralBaseline()
    train(model, data, epochs=20, seed=0)
    result = evaluate(model, test)
    assert result["subject_acc"] > 0.8  # chance is about 0.27
    assert result["iou_m"] > 0.3        # "move nothing" scores about 0.005
