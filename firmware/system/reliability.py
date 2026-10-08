import time
class RetryPolicy:
    def execute(self, func, retries=3):
        for _ in range(retries):
            try: return func()
            except: pass
        raise RuntimeError("Max retries exceeded")
