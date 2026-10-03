from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens

SUITES = [
    ("Standard Benchmark", "conversations.json"),
    ("Long-Context Stress Benchmark", "advanced_long_context.json"),
]
HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]
JUDGE_PROMPT = (
    "Chấm câu trả lời của trợ lý từ 0 đến 10.\n"
    "Tiêu chí: đúng các fact mong đợi, ngắn gọn, rõ ràng, không bịa.\n"
    "Câu hỏi: {question}\nFact mong đợi: {expected}\nCâu trả lời: {answer}\n"
    "Chỉ trả về một con số."
)


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a dataset file; a single conversation object is wrapped into a list."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _found(answer: str, expected: list[str]) -> int:
    lowered = answer.casefold()
    return sum(1 for item in expected if item.casefold() in lowered)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    found = _found(answer, expected)
    if found == len(expected):
        return 1.0
    return 0.5 if found else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    - 70% coverage: share of expected facts present (finer-grained than recall_points)
    - 15% concise: full marks up to 60 tokens, decreasing after that
    - 15% structured: answers with bullet lines, matching the user's preferred style
    """

    if not answer.strip():
        return 0.0
    coverage = _found(answer, expected) / len(expected) if expected else 1.0
    tokens = estimate_tokens(answer)
    concise = 1.0 if tokens <= 60 else max(0.0, 1 - (tokens - 60) / 120)
    structured = 1.0 if re.search(r"^\s*[-*•]\s", answer, re.MULTILINE) else 0.0
    return round(0.7 * coverage + 0.15 * concise + 0.15 * structured, 3)


def judge_quality(judge_model, question: str, answer: str, expected: list[str]) -> float | None:
    """LLM-as-judge score in [0, 1] (live mode only). None when the judge reply is unusable."""

    try:
        reply = judge_model.invoke(
            JUDGE_PROMPT.format(question=question, expected=", ".join(expected), answer=answer)
        )
    except Exception:
        return None
    match = re.search(r"\d+(?:\.\d+)?", getattr(reply, "text", None) or str(reply.content))
    return min(float(match.group()), 10.0) / 10 if match else None


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, judge_model=None
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Conversation turns run in thread `<conv id>`; every recall question runs in its
    own fresh thread so only persistent memory can answer it. Token columns count
    the conversation threads only, not the recall probes.
    """

    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    users = sorted({conv["user_id"] for conv in conversations})
    size_before = sum(memory_size(user) for user in users)

    agent_tokens = prompt_tokens = compactions = 0
    recalls: list[float] = []
    qualities: list[float] = []

    for conv in conversations:
        user_id, thread_id = conv["user_id"], conv["id"]
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, probe in enumerate(conv.get("recall_questions", [])):
            question, expected = probe["question"], probe["expected_contains"]
            answer = agent.reply(user_id, f"{thread_id}::recall::{index}", question)["response"]
            recalls.append(recall_points(answer, expected))
            score = judge_quality(judge_model, question, answer, expected) if judge_model else None
            qualities.append(score if score is not None else heuristic_quality(answer, expected))

    size_after = sum(memory_size(user) for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recalls) / len(recalls) if recalls else 0.0,
        response_quality=sum(qualities) / len(qualities) if qualities else 0.0,
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render rows as a GitHub-flavoured markdown table."""

    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github", disable_numparse=True)
    except ImportError:
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def _percent_change(new: int, old: int) -> str:
    return f"{(new - old) / old:+.0%}" if old else "n/a"


def main() -> None:
    """Run the Standard and Long-Context Stress suites for Baseline vs Advanced."""

    parser = argparse.ArgumentParser(description="Benchmark Baseline vs Advanced memory agents.")
    parser.add_argument("--live", action="store_true", help="call the real LLM configured in .env")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    judge_model = None
    if args.live:
        from model_provider import build_chat_model, has_credentials

        if not has_credentials(config.model):
            parser.error(
                f"--live needs credentials for provider '{config.model.provider}'. "
                "Set LLM_PROVIDER / LLM_MODEL and the matching API key in .env."
            )
        judge_model = build_chat_model(config.judge_model) if has_credentials(config.judge_model) else None

    print(f"Mode: {'live' if args.live else 'offline (deterministic)'} | "
          f"compact threshold={config.compact_threshold_tokens} tokens, keep={config.compact_keep_messages} messages\n")

    for title, filename in SUITES:
        conversations = load_conversations(config.data_dir / filename)

        # Fresh, isolated state per mode and suite so results are reproducible run after run
        # and an offline run never wipes the state of a live run in progress.
        mode_dir = "live" if args.live else "offline"
        suite_state = config.state_dir / "benchmark" / mode_dir / Path(filename).stem
        shutil.rmtree(suite_state, ignore_errors=True)
        suite_config = dataclasses.replace(config, state_dir=suite_state)

        rows = [
            run_agent_benchmark(
                "Baseline", BaselineAgent(suite_config, force_offline=not args.live), conversations, suite_config, judge_model
            ),
            run_agent_benchmark(
                "Advanced", AdvancedAgent(suite_config, force_offline=not args.live), conversations, suite_config, judge_model
            ),
        ]
        baseline, advanced = rows
        turns = sum(len(conv["turns"]) for conv in conversations)
        questions = sum(len(conv.get("recall_questions", [])) for conv in conversations)

        print(f"## {title}")
        print(f"{len(conversations)} conversation(s), {turns} turns, {questions} recall questions\n")
        print(format_rows(rows))
        print(
            f"\nAdvanced vs Baseline: prompt tokens {_percent_change(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
            f"agent tokens {_percent_change(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
            f"recall {advanced.recall_score - baseline.recall_score:+.2f}\n"
        )


if __name__ == "__main__":
    main()
