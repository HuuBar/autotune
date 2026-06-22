import threading


class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_tokens(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_summary(self) -> dict:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens
        }

    def reset(self):
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0


global_token_tracker = TokenTracker()
