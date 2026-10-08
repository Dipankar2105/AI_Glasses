class SystemRuntime:
    def __init__(self): self.state = "BOOT"
    def boot(self): self.state = "READY"
    def process_event(self, event):
        if event == "SHUTDOWN": self.state = "SHUTDOWN"
