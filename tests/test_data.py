import numpy as np

from nsrmp.data import gen_scene


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_scene_has_requested_objects_inside_unit_square_without_overlap():
    rng = np.random.RandomState(0)
    for n in (3, 7, 10):
        objs = gen_scene(rng, n)
        assert len(objs) == n
        for o in objs:
            x1, y1, x2, y2 = o.box
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
            assert 0.08 - 1e-9 <= x2 - x1 <= 0.12 + 1e-9
            assert 0.08 - 1e-9 <= y2 - y1 <= 0.12 + 1e-9
        for i in range(n):
            for j in range(i + 1, n):
                assert not overlaps(objs[i].box, objs[j].box)


def test_move_left_puts_subject_just_left_of_reference_keeping_its_size():
    from nsrmp.data import apply_action, GAP
    subj, ref = (0.7, 0.1, 0.8, 0.2), (0.5, 0.5, 0.6, 0.62)
    new = apply_action("MoveLeft", subj, ref)
    # right edge sits GAP to the left of the reference, vertically centred on it
    assert np.allclose(new, (0.5 - GAP - 0.1, 0.56 - 0.05, 0.5 - GAP, 0.56 + 0.05))


def test_move_front_goes_to_larger_y_and_behind_to_smaller_y():
    from nsrmp.data import apply_action, GAP
    subj, ref = (0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)
    front = apply_action("MoveFront", subj, ref)
    behind = apply_action("MoveBehind", subj, ref)
    assert np.allclose(front, (0.5, 0.6 + GAP, 0.6, 0.7 + GAP))
    assert np.allclose(behind, (0.5, 0.5 - GAP - 0.1, 0.6, 0.5 - GAP))


def test_move_right_is_clipped_to_the_unit_square_without_changing_size():
    from nsrmp.data import apply_action
    new = apply_action("MoveRight", (0.1, 0.1, 0.2, 0.2), (0.85, 0.5, 0.95, 0.6))
    assert np.allclose(new, (0.9, 0.5, 1.0, 0.6))


def test_relations_compare_box_centres():
    from nsrmp.data import holds
    a, b = (0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)
    assert holds("left", a, b) and not holds("right", a, b)
    assert holds("behind", a, b) and not holds("front", a, b)
    assert holds("front", b, a) and holds("right", b, a)


def test_every_simple_referring_expression_matches_exactly_one_object():
    from nsrmp.data import make_dataset, COLOURS, SHAPES
    for ex in make_dataset(300, seed=1, p_complex=0.0):
        words = ex.text.split()
        # "put <subject> <action phrase> <reference>": no relations inside simple NPs
        assert "which" not in words and ex.n_steps == 1
        for np_words in _simple_noun_phrases(ex.text):
            matches = [o for o in ex.objs if _matches(o, np_words)]
            assert len(matches) == 1, (ex.text, np_words)


def _simple_noun_phrases(text):
    """Split 'put the red cube to the left of the lego' into ['red','cube'] and ['lego']."""
    body = text.replace("put ", "")
    for phrase in ("to the left of", "to the right of", "in front of", "behind"):
        if phrase in body:
            left, right = body.split(phrase)
            return [left.split()[1:], right.split()[1:]]
    raise AssertionError(text)


def _matches(obj, np_words):
    from nsrmp.data import COLOURS, SHAPES
    ok = True
    for w in np_words:
        if w in COLOURS:
            ok &= obj.colour == w
        elif w in SHAPES:
            ok &= obj.shape == w
    return ok


def test_complex_instructions_pick_out_exactly_one_object_through_their_relation():
    from nsrmp.data import make_dataset, crisp_select
    exs = make_dataset(300, seed=2, p_complex=1.0)
    assert all("which is" in ex.text for ex in exs)
    for ex in exs:
        subj_prog, ref_prog = ex.program[0][2], ex.program[0][3]
        assert crisp_select(subj_prog, ex.objs).tolist() == [ex.subjects[0]]
        assert crisp_select(ref_prog, ex.objs).tolist() == [ex.refs[0]]


def test_gold_final_state_moves_only_the_subject_next_to_the_reference():
    from nsrmp.data import make_dataset, apply_action
    for ex in make_dataset(200, seed=3):
        s, r = ex.subjects[0], ex.refs[0]
        assert s != r
        action = ex.program[0][1]
        expected = apply_action(action, ex.objs[s].box, ex.objs[r].box)
        assert np.allclose(ex.final_boxes[s], expected)
        for i, o in enumerate(ex.objs):
            if i != s:
                assert np.allclose(ex.final_boxes[i], o.box)


def test_gold_program_of_simple_instruction_is_a_nested_move_tuple():
    from nsrmp.data import make_dataset
    ex = make_dataset(1, seed=4, p_complex=0.0)[0]
    (move,) = ex.program
    assert move[0] == "Move" and move[1] in ("MoveLeft", "MoveRight", "MoveFront", "MoveBehind")
    assert move[2][0] == "Unique" and move[2][1][0] == "Filter"
    assert move[2][1][1][0] == "Filter" and move[2][1][1][1] == ("Scene",)


def test_features_are_projected_one_hots_plus_small_noise():
    from nsrmp.data import make_dataset, feature_matrix, COLOURS, SHAPES
    ex = make_dataset(1, seed=5)[0]
    P = feature_matrix()
    o = ex.objs[0]
    onehot = np.zeros(9)
    onehot[COLOURS.index(o.colour)] = 1
    onehot[6 + SHAPES.index(o.shape)] = 1
    assert ex.feats.shape == (len(ex.objs), 32)
    resid = ex.feats[0] - P @ onehot
    assert 0.03 < resid.std() < 0.3  # sigma = 0.1


def test_multi_step_instruction_applies_steps_in_order_on_the_updated_scene():
    from nsrmp.data import make_dataset, crisp_select, apply_action
    for ex in make_dataset(100, seed=6, n_steps=(3, 3)):
        assert ex.text.count(" then ") == 2 and len(ex.program) == 3
        boxes = [o.box for o in ex.objs]
        for k, move in enumerate(ex.program):
            objs_now = [o.__class__(o.colour, o.shape, tuple(b)) for o, b in zip(ex.objs, boxes)]
            s = crisp_select(move[2], objs_now).tolist()
            r = crisp_select(move[3], objs_now).tolist()
            assert s == [ex.subjects[k]] and r == [ex.refs[k]]
            boxes[s[0]] = apply_action(move[1], boxes[s[0]], boxes[r[0]])
        assert np.allclose(ex.final_boxes, np.array(boxes))


def test_feature_noise_level_can_be_raised_without_changing_the_scenes():
    from nsrmp.data import make_dataset, feature_matrix, COLOURS, SHAPES
    clean, noisy = make_dataset(20, seed=5), make_dataset(20, seed=5, feature_noise=0.5)
    P = feature_matrix()
    resid = []
    for a, b in zip(clean, noisy):
        assert a.text == b.text and np.allclose(a.final_boxes, b.final_boxes)
        for o, f in zip(b.objs, b.feats):
            onehot = np.zeros(9)
            onehot[COLOURS.index(o.colour)] = 1
            onehot[6 + SHAPES.index(o.shape)] = 1
            resid.append(f - P @ onehot)
    assert 0.45 < np.std(resid) < 0.55
