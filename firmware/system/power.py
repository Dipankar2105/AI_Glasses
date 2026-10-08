class PowerManager:
    def __init__(self): self.state = "ACTIVE"
    def check_battery(self, level):
        if level < 10: self.state = "CRITICAL"
        elif level < 20: self.state = "LOW_POWER"
        else: self.state = "ACTIVE"
        return self.state
