"""Rule-based semantic parser: instruction text -> program.

Paper link: Sec. III-A (Language Parser). The paper's lexer/parser is *given* when training with
programs (an ablation in their repo); we do the same with a small deterministic recursive-descent
parser over the known keywords. Grammar:

    instruction := step ("," "then" step)*
    step        := "put" NP ACTION_PHRASE NP
    NP          := "the" [COLOUR] (SHAPE | "object") ["which" "is" RELATION_PHRASE SIMPLE_NP]
"""
from nsrmp.data import ACTIONS, ANY, COLOURS, RELATIONS, REL_PHRASE, SHAPES, noun_phrase_program


def split_steps(text):
    """The 'given splitter' also used by the baseline."""
    return [clause.strip(" ,") for clause in text.split(" then ")]


def _match_phrase(tokens, pos):
    """Return (relation, new_pos) if a relation phrase starts at tokens[pos]."""
    for rel, phrase in REL_PHRASE.items():
        words = phrase.split()
        if tokens[pos:pos + len(words)] == words:
            return rel, pos + len(words)
    raise ValueError(f"expected a relation phrase at word {pos}: {' '.join(tokens)}")


def _parse_simple_np(tokens, pos):
    if tokens[pos:pos + 1] != ["the"]:
        raise ValueError(f"expected 'the' at word {pos}: {' '.join(tokens)}")
    pos += 1
    colour = ANY
    if pos < len(tokens) and tokens[pos] in COLOURS:
        colour, pos = tokens[pos], pos + 1
    if pos < len(tokens) and tokens[pos] in SHAPES:
        shape = tokens[pos]
    elif pos < len(tokens) and tokens[pos] == "object":
        shape = ANY
    else:
        raise ValueError(f"expected a shape at word {pos}: {' '.join(tokens)}")
    return colour, shape, pos + 1


def _parse_np(tokens, pos):
    colour, shape, pos = _parse_simple_np(tokens, pos)
    if tokens[pos:pos + 2] != ["which", "is"]:
        return noun_phrase_program(colour, shape), pos
    rel, pos = _match_phrase(tokens, pos + 2)
    a_colour, a_shape, pos = _parse_simple_np(tokens, pos)
    anchor = noun_phrase_program(a_colour, a_shape)
    return noun_phrase_program(colour, shape, source=("Relate", anchor, rel)), pos


def _parse_step(clause):
    tokens = clause.lower().replace(",", " ").split()
    if not tokens or tokens[0] != "put":
        raise ValueError(f"a step must start with 'put': {clause!r}")
    subject, pos = _parse_np(tokens, 1)
    rel, pos = _match_phrase(tokens, pos)
    reference, pos = _parse_np(tokens, pos)
    if pos != len(tokens):
        raise ValueError(f"unexpected words after the instruction: {clause!r}")
    return ("Move", ACTIONS[RELATIONS.index(rel)], subject, reference)


def parse(text):
    """Return a tuple with one Move program per step."""
    return tuple(_parse_step(clause) for clause in split_steps(text))


def _np_roles(tokens, pos):
    """Word groups of one noun phrase: (head words, relation word, anchor words), new_pos.
    The relation and anchor groups are empty when the phrase has no "which is" clause."""
    start = pos
    _, _, pos = _parse_simple_np(tokens, pos)
    head = tokens[start:pos]
    if tokens[pos:pos + 2] != ["which", "is"]:
        return (head, [], []), pos
    rel, pos = _match_phrase(tokens, pos + 2)
    anchor_start = pos
    _, _, pos = _parse_simple_np(tokens, pos)
    return (head, [rel], tokens[anchor_start:pos]), pos


def parse_roles(clause):
    """The same parse as `_parse_step`, returned as word groups instead of a program (used by
    the parse-matched neural baseline). Relation/action phrases are reduced to their key word."""
    tokens = clause.lower().replace(",", " ").split()
    if not tokens or tokens[0] != "put":
        raise ValueError(f"a step must start with 'put': {clause!r}")
    subject, pos = _np_roles(tokens, 1)
    action, pos = _match_phrase(tokens, pos)
    reference, pos = _np_roles(tokens, pos)
    if pos != len(tokens):
        raise ValueError(f"unexpected words after the instruction: {clause!r}")
    return {"subject": subject, "reference": reference, "action": [action]}
