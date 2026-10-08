from firmware.interaction.gestures import GestureRecognizer
def test_p8():
    g = GestureRecognizer()
    assert g.process_imu(15,0,0) == "NOD"
    assert g.process_touch("DOUBLE_TAP") == "REPEAT"
test_p8()
