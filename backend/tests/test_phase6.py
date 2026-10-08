from backend.ai.ocr_processor import TTSEngine, ReadAloudController
class MockOCRRes:
    def __init__(self, t): self.full_text = t
def test_p6():
    c = ReadAloudController(TTSEngine())
    assert c.process_ocr(MockOCRRes("Hello")) == "Audio(Hello)"
    assert c.process_ocr(MockOCRRes("Hello")) is None # dedup
test_p6()
