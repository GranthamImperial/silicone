import pytest


@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.parametrize("entry_point", ("silicone-explore-quantiles-rolling-windows",))
def test_silicone_explore_quantiles_rolling_windows(entry_point, script_runner):
    res = script_runner.run(entry_point, "--help")
    assert res.success
    assert res.stdout
    assert res.stderr == ""
