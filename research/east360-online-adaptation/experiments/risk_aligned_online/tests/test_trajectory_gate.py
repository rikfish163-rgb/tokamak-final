import copy

from experiments.risk_aligned_online.trajectory_keys_20261028.run import screen


def fixture():
    risk = {part: {mode: {'bag': value, 'local': value} for mode, value in
                  (('memory', .2), ('off', .3), ('reverse', .25))} for part in ('all', 'D', 'S')}
    forecast = {'mean': .5, 'persistence_mean': .6, 'improved': 90, 'safe_improved': 50}
    control = copy.deepcopy(risk)
    control['all']['memory'] = {'bag': .28, 'local': .28}
    return risk, forecast, control


def test_all_preregistered_conditions_required():
    risk, forecast, control = fixture()
    assert screen(risk, forecast, control)['eligible']
    risk['S']['memory']['bag'] = .31
    assert not screen(risk, forecast, control)['eligible']


def test_reversed_feedback_and_safe_forecast_are_not_hidden_by_averages():
    risk, forecast, control = fixture()
    risk['all']['reverse']['local'] = .19
    assert not screen(risk, forecast, control)['eligible']
    risk, forecast, control = fixture()
    forecast['safe_improved'] = 40
    assert not screen(risk, forecast, control)['eligible']
