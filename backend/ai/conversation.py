class ConversationManager:
    def __init__(self):
        self.context = []
    def add_context(self, text):
        self.context.append(text)
        if len(self.context) > 10: self.context.pop(0)
    def respond(self, query):
        return f"MockResponse to {query}"
