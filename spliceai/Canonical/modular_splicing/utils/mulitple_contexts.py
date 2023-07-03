class MultipleContexts:
    def __init__(self, contexts):
        self.contexts = contexts

    def __enter__(self):
        for c in self.contexts:
            c.__enter__()
        return self.contexts

    def __exit__(self, *args):
        for c in self.contexts:
            c.__exit__(*args)
