from backend.ai.conversation import ConversationManager
def test_p7():
    c = ConversationManager()
    c.add_context("vision data")
    assert "MockResponse" in c.respond("What is this?")
test_p7()
