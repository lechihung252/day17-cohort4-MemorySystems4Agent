from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    is_recall_question,
    merge_profile,
)
from model_provider import build_chat_model, has_credentials, read_live_turn

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Bạn chỉ nhớ những gì người dùng nói trong cuộc trò chuyện hiện tại. "
    "Nếu không có thông tin trong cuộc trò chuyện này, hãy nói rõ là bạn không biết."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Every thread keeps its full message history, re-sent on every turn.
    - No `User.md`, no compaction: a new thread id starts from zero.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and has_credentials(self.config.model):
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:  # missing SDK / bad credentials -> stay offline
                print(f"[baseline] live mode unavailable, using offline: {exc}", file=sys.stderr)

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for one turn.

        `user_id` is accepted for interface parity with the advanced agent but
        deliberately unused: the baseline has no per-user memory.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # The whole thread history is the prompt context on every turn.
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        response = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": response})
        return self._record_turn(session, message, response, prompt_tokens, mode="offline")

    def _offline_response(self, session: SessionState, message: str) -> str:
        """Answer recall questions only from what was said in this thread."""

        if not is_recall_question(message):
            return "Đã ghi nhận."
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts = merge_profile(facts, extract_profile_updates(item["content"]))
        return answer_from_facts(message, facts)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}},
        )
        response, input_tokens, _ = read_live_turn(result)
        if input_tokens is None:
            input_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
                estimate_tokens(m["content"]) for m in session.messages
            )
        session.messages.append({"role": "assistant", "content": response})
        return self._record_turn(session, message, response, input_tokens, mode="live")

    def _record_turn(
        self, session: SessionState, message: str, response: str, prompt_tokens: int, mode: str
    ) -> dict[str, Any]:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "mode": mode,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: `create_agent` + `InMemorySaver`, no tools, no long-term memory."""

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
