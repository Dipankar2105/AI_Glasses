from firmware.system.reliability import RetryPolicy
def test_p11():
    r = RetryPolicy()
    f = lambda: True
    assert r.execute(f) is True
test_p11()
