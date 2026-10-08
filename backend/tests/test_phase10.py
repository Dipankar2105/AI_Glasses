from firmware.system.runtime import SystemRuntime
def test_p10():
    s = SystemRuntime()
    s.boot()
    assert s.state == "READY"
test_p10()
