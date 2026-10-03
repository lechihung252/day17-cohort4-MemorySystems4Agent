from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FACT_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    combine_confidence,
    estimate_tokens,
    extract_profile_candidates,
    is_recall_question,
)
from model_provider import build_chat_model, has_credentials, read_live_turn

try:  # live mode only; must be module-level so LangChain can resolve the tool/prompt type hints
    from langchain.agents.middleware import ModelRequest
    from langchain.tools import ToolRuntime
except ImportError:
    ModelRequest = ToolRuntime = None

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn về người dùng (User.md bên dưới). "
    "Dùng nó để cá nhân hóa câu trả lời và tuân theo style trả lời người dùng thích. "
    "Khi người dùng cung cấp fact ổn định mới (tên, nơi ở, nghề nghiệp, sở thích, style) "
    "hoặc đính chính fact cũ, gọi tool `save_user_fact`. Không lưu câu hỏi, câu đùa hay nơi chỉ ghé qua."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. within-session memory: recent messages of the thread (CompactMemoryManager)
    2. persistent `User.md`: stable facts shared by every thread of a user
    3. compact memory: older messages folded into a summary once a thread is too long
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        # Low-confidence facts waiting for another mention: user_id -> {(key, value): confidence}
        self.pending_facts: dict[str, dict[tuple[str, str], float]] = {}

        self.langchain_agent = None
        if not force_offline and has_credentials(self.config.model):
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:  # missing SDK / bad credentials -> stay offline
                print(f"[advanced] live mode unavailable, using offline: {exc}", file=sys.stderr)

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live mode (real LLM) and the deterministic offline mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> list[str]:
        """Steps 1-2: extract facts, persist the confident ones, park the rest as pending.

        A pending fact is promoted to User.md once repeated mentions push its combined
        confidence over `min_fact_confidence` (e.g. two hedged 0.5 mentions -> 0.75).
        """

        threshold = self.config.min_fact_confidence
        pending = self.pending_facts.setdefault(user_id, {})
        saved: list[str] = []
        for candidate in extract_profile_candidates(message):
            slot = (candidate.key, candidate.value)
            confidence = combine_confidence(pending.get(slot, 0.0), candidate.confidence)
            if confidence < threshold:
                pending[slot] = confidence
                continue
            pending.pop(slot, None)
            if self.profile_store.upsert_fact(user_id, candidate.key, candidate.value) and candidate.key not in saved:
                saved.append(candidate.key)
        return saved

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        saved = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message)
        if saved and not is_recall_question(message):
            response = "Đã ghi nhớ: " + ", ".join(FACT_LABELS[key].lower() for key in saved) + "."

        self.compact_memory.append(thread_id, "assistant", response)
        return self._record_turn(thread_id, message, response, prompt_tokens, saved, mode="offline")

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: User.md + compact summary + kept recent messages."""

        ctx = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(ctx["summary"])
            + sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from `User.md`, which survives across threads."""

        if not is_recall_question(message):
            return "Đã ghi nhận."
        return answer_from_facts(message, self.profile_store.facts(user_id))

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Deterministic extraction still runs so persistence never depends on the LLM calling a tool.
        saved = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}},
            context=context,
        )
        response, input_tokens, _ = read_live_turn(result)
        if input_tokens is None:
            input_tokens = estimate_tokens(ADVANCED_SYSTEM_PROMPT) + self._estimate_prompt_context_tokens(
                user_id, thread_id
            )

        # Mirror into compact memory so compaction counts stay comparable with offline mode.
        self.compact_memory.append(thread_id, "assistant", response)
        return self._record_turn(thread_id, message, response, input_tokens, saved, mode="live")

    def _record_turn(
        self, thread_id: str, message: str, response: str, prompt_tokens: int, saved: list[str], mode: str
    ) -> dict[str, Any]:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "mode": mode,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": saved,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + profile prompt + summarization."""

        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
        from langchain.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        store = self.profile_store
        model = build_chat_model(self.config.model)

        @tool
        def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the current user's long-term profile (User.md)."""
            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Save or correct one stable fact about the user in User.md.

            key: one of name, location, profession, interests, drink, food, pet, style.
            """
            changed = store.upsert_fact(runtime.context.user_id, key.strip().lower(), value.strip())
            return "saved" if changed else "unchanged"

        @dynamic_prompt
        def profile_prompt(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id)
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_profile>\n{profile}\n</user_profile>"

        return create_agent(
            model=model,
            tools=[read_user_profile, save_user_fact],
            middleware=[
                profile_prompt,
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
