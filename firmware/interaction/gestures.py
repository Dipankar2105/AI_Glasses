class GestureRecognizer:
    def __init__(self): self.last_action = None
    def process_imu(self, x, y, z):
        if x > 10: return "NOD"
        return "STABLE"
    def process_touch(self, action):
        if action == "DOUBLE_TAP": return "REPEAT"
        return "NONE"
