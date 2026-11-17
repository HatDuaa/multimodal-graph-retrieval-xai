"""Reading NativE's printed link-prediction metrics from a training log."""
import pytest

from scripts.parse_native_log import parse_command, parse_link_prediction

LOG = """$ python -u run_adv_wgan_gp.py -dataset=MKG-W -seed=42
Epoch 9 | D loss: 16.695446, G loss 121.940675: 100%|##########| 10/10 [00:39<00:00,  3.95s/it]
Test Head: 1 2 3 4
Test Tail: 1 2 3 4
metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1
l(filter):\t\t 0.019992 \t 2367.424561 \t 0.040711 \t 0.019420 \t 0.005147
r(filter):\t\t 0.294365 \t 279.227173 \t 0.437997 \t 0.319841 \t 0.218297
averaged(filter):\t 0.157178 \t 1323.325928 \t 0.239354 \t 0.169630 \t 0.111722


"""


def test_parses_head_tail_and_averaged_rows():
    metrics = parse_link_prediction(LOG)
    assert metrics["averaged"] == {"mrr": 0.157178, "mr": 1323.325928, "hits@10": 0.239354, "hits@3": 0.16963,
                                   "hits@1": 0.111722}
    assert metrics["head"]["mrr"] == 0.019992 and metrics["tail"]["hits@1"] == 0.218297


def test_an_unfinished_log_is_an_error():
    with pytest.raises(ValueError, match="averaged"):
        parse_link_prediction(LOG.split("averaged")[0])


def test_reads_the_driver_command_line():
    assert parse_command(LOG) == "python -u run_adv_wgan_gp.py -dataset=MKG-W -seed=42"
    assert parse_command("no command") == ""


def test_aggregate_is_mean_and_sample_std():
    from scripts.parse_native_log import aggregate
    runs = [dict.fromkeys(("mrr", "mr", "hits@10", "hits@3", "hits@1"), v) for v in (0.1, 0.2, 0.3)]
    out = aggregate(runs)
    assert out["mrr"]["mean"] == pytest.approx(0.2) and out["mrr"]["std"] == pytest.approx(0.1)
    assert out["hits@1"]["n_runs"] == 3
