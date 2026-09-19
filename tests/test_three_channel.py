import numpy as np

from src.graph.soft_match import channel_scores, fuse_three_channels, instance_parts, zscore_channel


def test_zscore_nan_constant_and_single():
    np.testing.assert_allclose(zscore_channel(np.array([1.0, np.nan, 3.0])), [-1.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(zscore_channel(np.array([2.0, 2.0])), [0.0, 0.0])
    np.testing.assert_allclose(zscore_channel(np.array([np.nan, 2.0])), [0.0, 0.0])


def test_fuse_inactive_and_one_active():
    clip = np.array([1.0, 2.0, 3.0])
    inactive = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_allclose(fuse_three_channels(clip, inactive, inactive, 0.4, 0.2), zscore_channel(clip) * 0.4)
    obj = np.array([0.1, 0.4, 0.9])
    np.testing.assert_allclose(fuse_three_channels(clip, obj, inactive, 0.5, 0.0),
                               fuse_three_channels(clip, obj, inactive, 0.5, 1.0))
    np.testing.assert_allclose(fuse_three_channels(clip, obj, inactive, 1.0, 0.3), zscore_channel(clip))


def test_channel_scores_temperature_approaches_max():
    query = np.array([[1.0, 0.0]])
    images = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    assert channel_scores(query, images, 1e-5)[0] > 0.99


def test_instance_parts_keep_duplicates():
    graph = {"objects": [{"id": 0, "name": "person"}, {"id": 1, "name": "person"}],
             "relations": [{"subject": 0, "predicate": "on", "object": 1}]}
    assert instance_parts(graph) == (["person", "person"], ["person on person"])
