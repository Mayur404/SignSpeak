"""Word / sentence composition from committed signs."""
from __future__ import annotations

from dataclasses import dataclass, field

from .labels import BACKSPACE, CUSTOM_WORD_PREFIX, PERIOD, SPACE


@dataclass
class CommitResult:
    kind: str  # "letter" | "word" | "backspace" | "space" | "period"
    word: str
    transcript: str
    speak_word: str | None = None
    speak_sentence: str | None = None
    changed: bool = True


@dataclass
class TextComposer:
    tokens: list[str] = field(default_factory=list)  # finished words + "."
    word: str = ""

    # ---- rendering -------------------------------------------------------
    @staticmethod
    def _render(tokens: list[str]) -> str:
        out = ""
        for token in tokens:
            if token == PERIOD:
                out = out.rstrip() + PERIOD
            else:
                out = f"{out} {token}".strip()
        return out

    @property
    def transcript(self) -> str:
        return self._render(self.tokens)

    @property
    def sentence_tokens(self) -> list[str]:
        index = 0
        for i, token in enumerate(self.tokens):
            if token == PERIOD:
                index = i + 1
        return [t for t in self.tokens[index:] if t != PERIOD]

    @property
    def sentence(self) -> str:
        return self._render(self.sentence_tokens)

    @property
    def speak_text(self) -> str:
        parts = []
        if self.transcript:
            parts.append(self.transcript)
        if self.word:
            parts.append(self.word)
        return " ".join(parts).strip()

    # ---- actions ---------------------------------------------------------
    def commit(self, label: str) -> CommitResult:
        if label == BACKSPACE:
            changed = self.backspace()
            return CommitResult("backspace", self.word, self.transcript, changed=changed)
        if label == SPACE:
            return self.space()
        if label == PERIOD:
            return self.period()
        if label.startswith(CUSTOM_WORD_PREFIX):
            custom_word = label[len(CUSTOM_WORD_PREFIX) :].strip()
            if not custom_word:
                return CommitResult("word", self.word, self.transcript, changed=False)
            if self.word:
                self.tokens.append(self.word)
                self.word = ""
            self.tokens.append(custom_word)
            return CommitResult(
                "word", self.word, self.transcript, speak_word=custom_word
            )
        if len(label) == 1:
            self.word += label
            return CommitResult("letter", self.word, self.transcript)
        return CommitResult("letter", self.word, self.transcript, changed=False)

    def space(self) -> CommitResult:
        if not self.word:
            return CommitResult("space", "", self.transcript, changed=False)
        finished = self.word
        self.tokens.append(finished)
        self.word = ""
        return CommitResult(
            "space",
            "",
            self.transcript,
            speak_word=finished,
        )

    def period(self) -> CommitResult:
        if self.word:
            self.tokens.append(self.word)
            self.word = ""
        sentence = self.sentence
        if not sentence:
            return CommitResult("period", "", self.transcript, changed=False)
        self.tokens.append(PERIOD)
        return CommitResult(
            "period",
            "",
            self.transcript,
            speak_sentence=sentence,
        )

    def backspace(self) -> bool:
        if self.word:
            self.word = self.word[:-1]
            return True
        if self.tokens:
            self.tokens.pop()
            return True
        return False

    def delete_word(self) -> bool:
        """Remove the unfinished word, or the last completed transcript word."""
        if self.word:
            self.word = ""
            return True
        if not self.tokens:
            return False
        if self.tokens[-1] == PERIOD:
            self.tokens.pop()
        for index in range(len(self.tokens) - 1, -1, -1):
            if self.tokens[index] != PERIOD:
                del self.tokens[index]
                return True
        return False

    def clear_word(self) -> bool:
        changed = bool(self.word)
        self.word = ""
        return changed

    def clear_all(self) -> None:
        self.tokens.clear()
        self.word = ""

    def load(self, transcript: str, word: str = "") -> None:
        self.tokens = [token for token in transcript.replace(".", " . ").split() if token]
        self.word = word
