class ThermalManager:
    def check_temp(self, temp):
        if temp > 80: return "CRITICAL"
        if temp > 60: return "WARM"
        return "NORMAL"
