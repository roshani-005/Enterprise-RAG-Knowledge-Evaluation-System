from eval.run_eval import recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(['a', 'b', 'c'], {'b', 'c'}) == 1.0
    assert recall_at_k(['a'], {'b'}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(['x', 'y', 'z'], {'y'}) == 0.5
    assert reciprocal_rank(['x'], {'y'}) == 0.0
