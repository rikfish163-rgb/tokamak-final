def test_receipt_is_reduced_to_learning_episode_not_audit_fields():
    from experiments.risk_aligned_online.continual_episode_20261023.diagnose import receipt_episode
    receipt = dict(shot=20, count=256, end=270, label=0,
                   forecast_origins=[130,155,180,205], inner_loss=.1)
    assert receipt_episode(receipt) == dict(count=256,end=270,label=0)
    assert receipt['shot'] == 20
