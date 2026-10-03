from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import tempfile
import threading


def estimate_tokens(text: str) -> int:
    """Approximate token count: ~4 characters per token, 0 for empty text."""

    text = text.strip()
    if not text:
        return 0
    return math.ceil(len(text) / 4)


FACT_LINE = re.compile(r"^- (\w+): (.+)$", re.MULTILINE)

# Facts that accumulate over time instead of being replaced by the newest value.
MERGE_KEYS = ("style", "interests")


def _merge_values(old: str, new: str) -> str:
    """Union of comma-separated items, keeping the original order."""

    items = [item.strip() for item in old.split(",") if item.strip()]
    for item in new.split(","):
        item = item.strip()
        if item and item not in items:
            items.append(item)
    return ", ".join(items)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user).

    Layout: `<root_dir>/<user_id>/User.md`, facts stored as `- key: value` lines.

    Thread-safe: live agents run tool calls in parallel, so read-modify-write is
    serialized by a lock and files are replaced atomically (never seen half-written).
    """

    root_dir: Path
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w-]", "_", user_id.strip().lower()) or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return f"# User profile: {user_id}\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".User.md.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        with self._lock:
            content = self.read_text(user_id)
            if not search_text or search_text not in content:
                return False
            self.write_text(user_id, content.replace(search_text, replacement, 1))
            return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return {key: value.strip() for key, value in FACT_LINE.findall(self.read_text(user_id))}

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or update one fact. Returns False when nothing changed on disk.

        Corrections overwrite the old value (e.g. location Đà Nẵng -> Huế),
        except for MERGE_KEYS where new preferences are added to the old ones.
        Keys/values are normalized so every fact stays one parseable `- key: value` line
        (LLM tool calls may send "món ăn" as a key or multi-line values).
        """

        key = re.sub(r"\W+", "_", key.strip().lower()).strip("_")
        value = " ".join(value.split())
        if not key or not value:
            return False

        with self._lock:
            facts = self.facts(user_id)
            old = facts.get(key)
            if old is not None and key in MERGE_KEYS:
                value = _merge_values(old, value)
            if old == value:
                return False
            facts[key] = value
            lines = [f"# User profile: {user_id}", ""] + [f"- {k}: {v}" for k, v in facts.items()]
            self.write_text(user_id, "\n".join(lines) + "\n")
            return True


PLACES = ["Đà Nẵng", "Huế", "Hà Nội", "TP.HCM", "Sài Gòn", "Hải Phòng", "Cần Thơ", "Nha Trang"]
_PLACE = "|".join(map(re.escape, PLACES))

LOCATION_RE = re.compile(
    rf"(?:mình ở|đang ở|hiện ở|vẫn ở|sẽ ở|làm việc ở|chuyển (?:ra|vào|về|tới|đến))\s+({_PLACE})",
    re.IGNORECASE,
)
PROFESSION_RE = re.compile(r"(?:làm|chuyển sang|là)\s+([A-Za-z]+ engineer)", re.IGNORECASE)
NAME_RE = re.compile(r"tên(?: mình)? là\s+([^,.?!]+?)\s*(?:[,.?!]|$)", re.IGNORECASE)
DRINK_RE = re.compile(r"đồ uống yêu thích (?:là|của mình là)\s+([^,.?!]+)", re.IGNORECASE)
FOOD_RE = re.compile(r"món ăn yêu thích (?:là|của mình là)\s+([^,.?!]+)", re.IGNORECASE)
PET_RE = re.compile(r"nuôi (?:một )?(?:bé |con )?(\w+)(?: tên (\w+))?", re.IGNORECASE)

# "chứ không còn ở Đà Nẵng", "không phải nơi ở hiện tại" -> removed before matching.
NEGATION_RE = re.compile(r"(?:chứ )?không (?:còn|phải)[^,.;]*", re.IGNORECASE)
NOISE_MARKERS = ("đùa", "chỉ là nơi", "đừng lấy", "đừng nói")
QUESTION_WORDS = ("gì", "nào", "đâu", "ai")

STYLE_TRIGGERS = ("trả lời", "giải thích", "style")
STYLE_KEYWORDS = [
    ("3 bullet", "3 bullet"),
    ("ngắn gọn", "ngắn gọn"),
    ("bullet", "bullet"),
    ("ví dụ thực", "có ví dụ thực tế"),
    ("trade-off", "nhấn trade-off"),
]
INTEREST_TRIGGERS = ("thích", "quan tâm")
INTERESTS = ("Python", "AI", "MLOps", "RAG")

SIMPLE_PATTERNS = [
    ("location", LOCATION_RE),
    ("profession", PROFESSION_RE),
    ("name", NAME_RE),
    ("drink", DRINK_RE),
    ("food", FOOD_RE),
]


def _split_sentences(message: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", message.strip()) if s]


# --- Confidence threshold (bonus) -------------------------------------------------
# How reliable each extraction rule is when the sentence is a plain statement.
BASE_CONFIDENCE = {
    "name": 0.95,
    "drink": 0.9,
    "food": 0.9,
    "pet": 0.85,
    "location": 0.85,
    "profession": 0.85,
    "style": 0.8,
    "interests": 0.75,
}
# Plans, guesses and hypotheticals are not facts yet.
HEDGE_MARKERS = ("có lẽ", "hình như", "chắc là", "định ", "dự định", "sắp ", "sẽ ", "đang cân nhắc")
HEDGE_PENALTY = 0.35
# A leading "Nếu ..." makes the sentence hypothetical ("…nếu cần" at the end does not).
# Style preferences are naturally conditional ("Nếu giải thích, hãy trả lời ngắn gọn"), so they are exempt.
CONDITIONAL_PREFIX = "nếu "
# Explicit corrections / confirmations make a statement more reliable.
CONFIRM_MARKERS = ("đính chính", "nhớ là", "hiện tại", "bây giờ", "thực ra")
CONFIRM_BONUS = 0.1
DEFAULT_MIN_CONFIDENCE = 0.7


@dataclass(frozen=True)
class FactCandidate:
    key: str
    value: str
    confidence: float


def combine_confidence(previous: float, new: float) -> float:
    """Independent evidence: two 0.5 mentions -> 0.75. Used to promote repeated pending facts."""

    return 1 - (1 - previous) * (1 - new)


def _sentence_confidence(key: str, lowered: str) -> float:
    confidence = BASE_CONFIDENCE[key]
    hypothetical = lowered.lstrip().startswith(CONDITIONAL_PREFIX) and key != "style"
    if hypothetical or any(marker in lowered for marker in HEDGE_MARKERS):
        confidence -= HEDGE_PENALTY
    if any(marker in lowered for marker in CONFIRM_MARKERS):
        confidence += CONFIRM_BONUS
    return round(min(max(confidence, 0.0), 1.0), 2)


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Every fact found in the message, in order, with a confidence score.

    Per sentence: skip questions and noise, drop negated clauses, then run the
    patterns. Within a sentence the last match wins so
    "lúc đầu ở Huế, nhưng giờ ở Đà Nẵng" -> Đà Nẵng.
    """

    candidates: list[FactCandidate] = []
    for sentence in _split_sentences(message):
        lowered = sentence.lower()
        if sentence.endswith("?") or any(marker in lowered for marker in NOISE_MARKERS):
            continue

        cleaned = NEGATION_RE.sub("", sentence)
        lowered = cleaned.lower()
        found: list[tuple[str, str]] = []

        for key, pattern in SIMPLE_PATTERNS:
            matches = pattern.findall(cleaned)
            if matches:
                value = matches[-1].strip()
                if value.lower() not in QUESTION_WORDS:
                    found.append((key, value))

        pet = PET_RE.search(cleaned)
        if pet and pet.group(1).lower() not in QUESTION_WORDS:
            found.append(("pet", pet.group(1) + (f" tên {pet.group(2)}" if pet.group(2) else "")))

        if any(trigger in lowered for trigger in STYLE_TRIGGERS):
            style = [label for keyword, label in STYLE_KEYWORDS if keyword in lowered]
            if "3 bullet" in style and "bullet" in style:
                style.remove("bullet")
            if style:
                found.append(("style", ", ".join(style)))

        if any(trigger in lowered for trigger in INTEREST_TRIGGERS):
            interests = [item for item in INTERESTS if re.search(rf"\b{item}\b", cleaned)]
            if interests:
                found.append(("interests", ", ".join(interests)))

        candidates += [FactCandidate(key, value, _sentence_confidence(key, lowered)) for key, value in found]
    return candidates


def extract_profile_updates(message: str, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict[str, str]:
    """Stable profile facts whose confidence reaches `min_confidence` (later sentences win)."""

    updates: dict[str, str] = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence >= min_confidence:
            updates = merge_profile(updates, {candidate.key: candidate.value})
    return updates


def merge_profile(facts: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Apply updates to an in-memory fact dict with the same rules as `upsert_fact`."""

    merged = dict(facts)
    for key, value in updates.items():
        if key in MERGE_KEYS and key in merged:
            value = _merge_values(merged[key], value)
        merged[key] = value
    return merged


FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "interests": "Mối quan tâm",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "style": "Style trả lời",
}

# Question keywords -> which fact the user is asking about.
QUESTION_HINTS = [
    ("name", ("tên", "là ai")),
    ("location", ("ở đâu", "nơi ở", "còn ở", "đang ở")),
    ("profession", ("nghề",)),
    ("interests", ("quan tâm",)),
    ("drink", ("đồ uống",)),
    ("food", ("món ăn",)),
    ("pet", ("nuôi", "thú cưng")),
    ("style", ("style", "kiểu trả lời", "trả lời")),
]
RECALL_MARKERS = ("nhắc lại", "nhớ lại", "tóm tắt", "là gì", "là ai")


def is_recall_question(message: str) -> bool:
    lowered = message.lower()
    return message.strip().endswith("?") or any(marker in lowered for marker in RECALL_MARKERS)


def requested_fact_keys(question: str) -> list[str]:
    lowered = question.lower()
    return [key for key, hints in QUESTION_HINTS if any(hint in lowered for hint in hints)]


def answer_from_facts(question: str, facts: dict[str, str]) -> str:
    """Deterministic recall answer: one bullet per requested fact that is known."""

    keys = requested_fact_keys(question) or list(facts)
    known = [key for key in keys if key in facts]
    if not known:
        return "Mình chưa có thông tin này trong bộ nhớ hiện tại."
    lines = [f"- {FACT_LABELS[key]}: {facts[key]}" for key in known]
    missing = [FACT_LABELS[key].lower() for key in keys if key not in facts]
    if missing:
        lines.append(f"- Chưa rõ: {', '.join(missing)}")
    return "\n".join(lines)


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: one truncated line per message, newest `max_items` only."""

    lines = []
    for message in messages[-max_items:]:
        content = " ".join(message.get("content", "").split())
        if len(content) > 80:
            content = content[:80].rstrip() + "…"
        lines.append(f"- {message.get('role', 'user')}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the newest `keep_messages` in full; once the thread exceeds
    `threshold_tokens`, older messages are folded into a running summary.
    """

    threshold_tokens: int
    keep_messages: int
    summary_max_lines: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        ctx = self.context(thread_id)
        ctx["messages"].append({"role": role, "content": content})
        if self._context_tokens(ctx) > self.threshold_tokens and len(ctx["messages"]) > self.keep_messages:
            self._compact(ctx)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])

    def _context_tokens(self, ctx: dict[str, object]) -> int:
        return estimate_tokens(ctx["summary"]) + sum(estimate_tokens(m["content"]) for m in ctx["messages"])

    def _compact(self, ctx: dict[str, object]) -> None:
        messages = ctx["messages"]
        old, ctx["messages"] = messages[: -self.keep_messages], messages[-self.keep_messages :]
        # Append the new summary lines to the running summary, capped so it cannot grow forever.
        lines = ctx["summary"].splitlines() + summarize_messages(old, max_items=len(old)).splitlines()
        ctx["summary"] = "\n".join(lines[-self.summary_max_lines :])
        ctx["compactions"] += 1
