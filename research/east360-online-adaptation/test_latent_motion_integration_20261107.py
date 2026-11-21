import torch

from run_latent_motion_20261107 import context


def data():
    torch.manual_seed(131)
    return dict(context=torch.randn(3, 69),
                packs={k: dict(risk_hidden=torch.randn(3, 64)) for k in ("r", "p", "pp", "q")},
                labels=torch.tensor([0, 1, 0]))


def test_update_context_cannot_read_future_query_state_or_labels():
    d = data()
    expected = {k: context(d, k) for k in ("ordered", "stale", "swap")}
    d["packs"]["q"]["risk_hidden"].fill_(float("nan"))
    d["labels"][:] = 1-d["labels"]
    for kind, value in expected.items():
        torch.testing.assert_close(context(d, kind), value, rtol=0., atol=0.)


def test_ordered_context_uses_the_observed_release_state():
    d = data()
    expected = context(d, "ordered")
    d["packs"]["r"]["risk_hidden"] += 1
    assert not torch.equal(context(d, "ordered"), expected)
    torch.testing.assert_close(context(d, "summary"), torch.cat((d["context"], torch.zeros(3, 192)), 1))
