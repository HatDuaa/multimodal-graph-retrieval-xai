from src.data.train_groups import make_train_groups


def test_groups_are_deterministic_balanced_and_cover_train():
    ids = list(range(23))
    first = make_train_groups(ids, seed=7, dev_size=5, group_size=6)
    second = make_train_groups(reversed(ids), seed=7, dev_size=5, group_size=6)
    assert first == second
    dev = set(first["dev"])
    groups = [item for group in first["groups"] for item in group]
    assert len(dev) == 5
    assert not dev.intersection(groups)
    assert set(groups) == set(ids) - dev
    assert max(map(len, first["groups"])) - min(map(len, first["groups"])) <= 1


def test_zero_dev_distributes_all_train_images():
    result = make_train_groups(range(23), seed=7, dev_size=0, group_size=6)
    assert result["dev"] == []
    assert sorted(item for group in result["groups"] for item in group) == list(range(23))
    assert max(map(len, result["groups"])) - min(map(len, result["groups"])) <= 1
