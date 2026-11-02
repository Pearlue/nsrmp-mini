import pytest

from nsrmp.data import make_dataset
from nsrmp.parser import parse, parse_roles, split_steps


def test_simple_instruction_parses_to_the_expected_program():
    prog = parse("put the red cube to the right of the blue dice")
    expected = (("Move", "MoveRight",
                 ("Unique", ("Filter", ("Filter", ("Scene",), "red"), "cube")),
                 ("Unique", ("Filter", ("Filter", ("Scene",), "blue"), "dice"))),)
    assert prog == expected


def test_relative_subject_is_filtered_from_the_related_set():
    prog = parse("put the lego which is behind the green dice to the left of the red cube")
    subject = ("Unique", ("Filter", ("Filter",
               ("Relate", ("Unique", ("Filter", ("Filter", ("Scene",), "green"), "dice")), "behind"),
               "any"), "lego"))
    assert prog[0][1] == "MoveLeft" and prog[0][2] == subject
    assert prog[0][3] == ("Unique", ("Filter", ("Filter", ("Scene",), "red"), "cube"))


def test_relation_without_which_is_is_rejected_instead_of_guessed():
    # "in front of the red lego" here would have to be a relation on the subject; only
    # "which is <relation>" introduces one, so the extra phrase is an error.
    with pytest.raises(ValueError):
        parse("put the cube in front of the red lego in front of the dice")


def test_multi_step_instruction_is_split_on_then():
    text = "put the red cube behind the blue dice, then put the lego to the left of the cube"
    assert split_steps(text) == ["put the red cube behind the blue dice",
                                 "put the lego to the left of the cube"]
    assert [m[1] for m in parse(text)] == ["MoveBehind", "MoveLeft"]


def test_unknown_words_raise():
    with pytest.raises(ValueError):
        parse("push the purple cube behind the blue dice")


@pytest.mark.parametrize("kwargs", [
    dict(p_complex=0.0), dict(p_complex=1.0), dict(n_steps=(1, 4)),
])
def test_parser_recovers_the_gold_program_of_every_generated_template(kwargs):
    for ex in make_dataset(300, seed=7, n_objects=(3, 8), **kwargs):
        assert parse(ex.text) == ex.program, ex.text


def test_role_phrases_keep_the_same_words_the_program_uses():
    # Only the concepts the program contains: no "the", and "object" (the ANY wildcard) is dropped.
    roles = parse_roles("put the lego which is behind the green dice to the left of the red cube")
    assert roles == {"subject": (["lego"], ["behind"], ["green", "dice"]),
                     "reference": (["red", "cube"], [], []),
                     "action": ["left"]}
    other = parse_roles("put the cube in front of the blue object which is to the right of the dice")
    assert other["subject"] == (["cube"], [], [])
    assert other["reference"] == (["blue"], ["right"], ["dice"])
    assert other["action"] == ["front"]
