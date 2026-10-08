from firmware.system.power import PowerManager
from firmware.system.thermal import ThermalManager
def test_p9():
    p = PowerManager()
    assert p.check_battery(5) == "CRITICAL"
    assert ThermalManager().check_temp(85) == "CRITICAL"
test_p9()
