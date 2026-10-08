class TTSEngine:
    def process(self, text):
        return f"Audio({text})"
        
class ReadAloudController:
    def __init__(self, tts):
        self.tts = tts
        self.last_text = ""
        
    def process_ocr(self, ocr_result):
        if not ocr_result or not ocr_result.full_text:
            return None
        t = ocr_result.full_text.strip()
        if t == self.last_text or len(t) < 2:
            return None
        self.last_text = t
        return self.tts.process(t)
