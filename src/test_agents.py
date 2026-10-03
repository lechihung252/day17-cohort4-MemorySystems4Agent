from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points
from config import load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_candidates,
    extract_profile_updates,
)

LONG_TURN = "Mình kể thêm về pipeline MLOps, data drift, monitoring và chi phí GPU hằng ngày. " * 4


def make_config(tmp_path: Path):
    """Isolated config: state under tmp_path, tiny compact threshold so compaction happens fast."""

    return dataclasses.replace(
        load_config(),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=2,
    )


def make_agents(tmp_path: Path) -> tuple[BaselineAgent, AdvancedAgent]:
    config = make_config(tmp_path)
    return BaselineAgent(config, force_offline=True), AdvancedAgent(config, force_offline=True)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.file_size("dungct") == 0
    assert "dungct" in store.read_text("dungct")  # default profile when the file is missing

    path = store.write_text("dungct", "# User profile: dungct\n\n- name: DũngCT\n")
    assert path == tmp_path / "profiles" / "dungct" / "User.md"
    assert store.facts("dungct") == {"name": "DũngCT"}

    assert store.edit_text("dungct", "DũngCT", "DũngCT Stress")
    assert not store.edit_text("dungct", "không tồn tại", "x")
    assert store.facts("dungct")["name"] == "DũngCT Stress"
    assert store.file_size("dungct") == len(store.read_text("dungct").encode("utf-8"))

    assert "../" not in str(store.path_for("../../etc/passwd"))


def test_upsert_overwrites_corrections_and_merges_preferences(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.upsert_fact("u", "location", "Đà Nẵng")
    assert store.upsert_fact("u", "location", "Huế")
    assert store.facts("u")["location"] == "Huế"
    assert "Đà Nẵng" not in store.read_text("u")

    size = store.file_size("u")
    assert not store.upsert_fact("u", "location", "Huế")  # unchanged fact -> no rewrite
    assert store.file_size("u") == size

    store.upsert_fact("u", "style", "ngắn gọn")
    store.upsert_fact("u", "style", "nhấn trade-off")
    assert store.facts("u")["style"] == "ngắn gọn, nhấn trade-off"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Chào bạn, mình tên là DũngCT.", {"name": "DũngCT"}),
        ("giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.", {"location": "Huế"}),
        ("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.", {"profession": "MLOps engineer"}),
        ("Mình làm MLOps engineer chứ không còn là backend engineer nữa nhé.", {"profession": "MLOps engineer"}),
        ("Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng.", {"location": "Đà Nẵng"}),
        ("Mình nuôi một bé corgi tên Bơ.", {"pet": "corgi tên Bơ"}),
    ],
)
def test_extract_profile_updates_handles_corrections(message: str, expected: dict[str, str]) -> None:
    updates = extract_profile_updates(message)
    for key, value in expected.items():
        assert updates.get(key) == value


@pytest.mark.parametrize(
    "message",
    [
        "Bạn có thể nhắc lại tên mình không?",
        "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
        "Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.",
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại.",
        "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
        "Nếu sau này mình có nhắc lại Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại.",
    ],
)
def test_extract_ignores_questions_and_noise(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_compact_trigger(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)

    advanced.reply("u", "short", "Xin chào.")
    assert advanced.compaction_count("short") == 0  # short threads never compact

    for _ in range(6):
        advanced.reply("u", "long", LONG_TURN)
    ctx = advanced.compact_memory.context("long")
    assert advanced.compaction_count("long") >= 1
    assert len(ctx["messages"]) == 2  # only `keep_messages` stay verbatim
    assert ctx["summary"]


def test_summary_stays_bounded() -> None:
    memory = CompactMemoryManager(threshold_tokens=40, keep_messages=2, summary_max_lines=5)
    for i in range(50):
        memory.append("t", "user", f"tin nhắn {i} " + "x" * 120)
    assert memory.compaction_count("t") > 10
    assert len(memory.context("t")["summary"].splitlines()) <= 5


def test_cross_session_recall(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    facts = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
    ]
    for message in facts:
        baseline.reply("dungct", "session-1", message)
        advanced.reply("dungct", "session-1", message)

    question = "Mình tên gì và hiện tại đang ở đâu?"
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]

    assert "DũngCT" in advanced_answer and "Huế" in advanced_answer
    assert "Đà Nẵng" not in advanced_answer  # correction wins over the old fact
    assert "DũngCT" not in baseline_answer and "Huế" not in baseline_answer

    # A brand-new agent instance still remembers: User.md lives on disk.
    _, restarted = make_agents(tmp_path)
    assert "DũngCT" in restarted.reply("dungct", "session-3", question)["response"]


def test_baseline_remembers_within_same_thread(tmp_path: Path) -> None:
    baseline, _ = make_agents(tmp_path)
    baseline.reply("u", "t", "Mình tên là Lan.")
    assert "Lan" in baseline.reply("u", "t", "Mình tên gì?")["response"]
    assert baseline.compaction_count("t") == 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    stress = load_conversations(Path(__file__).resolve().parent.parent / "data" / "advanced_long_context.json")[0]

    for turn in stress["turns"]:
        baseline.reply(stress["user_id"], "stress", turn)
        advanced.reply(stress["user_id"], "stress", turn)

    assert advanced.compaction_count("stress") >= 2
    assert advanced.prompt_token_usage("stress") < baseline.prompt_token_usage("stress") * 0.5


def test_short_thread_costs_more_for_advanced(tmp_path: Path) -> None:
    """Trade-off: on short threads the User.md overhead makes Advanced more expensive."""

    config = dataclasses.replace(make_config(tmp_path), compact_threshold_tokens=10_000)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for message in ["Mình tên là DũngCT.", "Mình đang ở Huế.", "Món ăn yêu thích là mì Quảng."]:
        baseline.reply("u", "t", message)
        advanced.reply("u", "t", message)
    assert advanced.prompt_token_usage("t") > baseline.prompt_token_usage("t")


def test_scoring_helpers() -> None:
    assert recall_points("DũngCT ở Huế", ["DũngCT", "Huế"]) == 1.0
    assert recall_points("DũngCT ở Đà Nẵng", ["DũngCT", "Huế"]) == 0.5
    assert recall_points("Mình không biết.", ["DũngCT"]) == 0.0
    good = heuristic_quality("- Tên: DũngCT\n- Nơi ở: Huế", ["DũngCT", "Huế"])
    bad = heuristic_quality("Mình chưa có thông tin này.", ["DũngCT", "Huế"])
    assert good > 0.9 and bad < 0.3


# --- Bonus: confidence threshold -----------------------------------------------------

HEDGED_PROBES = [
    "Tháng sau có lẽ mình sẽ chuyển ra Hà Nội.",
    "Mình đang cân nhắc chuyển sang data engineer.",
    "Mình định nuôi một con mèo.",
    "Hình như món ăn yêu thích là phở, mình cũng không chắc.",
    "Nếu mình chuyển ra Hà Nội thì chắc sẽ thuê nhà gần hồ.",
    "Mình sắp chuyển sang làm data engineer, chưa chốt đâu.",
]


def seeded_advanced(tmp_path: Path, min_confidence: float = 0.7) -> AdvancedAgent:
    config = dataclasses.replace(make_config(tmp_path), min_fact_confidence=min_confidence)
    agent = AdvancedAgent(config, force_offline=True)
    for key, value in {"location": "Huế", "profession": "MLOps engineer", "food": "mì Quảng"}.items():
        agent.profile_store.upsert_fact("u", key, value)
    return agent


@pytest.mark.parametrize("message", HEDGED_PROBES)
def test_hedged_statements_do_not_overwrite_profile(tmp_path: Path, message: str) -> None:
    agent = seeded_advanced(tmp_path)
    before = agent.profile_store.facts("u")
    agent.reply("u", "t", message)
    assert agent.profile_store.facts("u") == before
    assert extract_profile_candidates(message)  # a fact was seen, just not trusted


@pytest.mark.parametrize("message", HEDGED_PROBES)
def test_without_threshold_hedged_statements_corrupt_profile(tmp_path: Path, message: str) -> None:
    """Ablation: threshold 0 shows the failure mode the bonus prevents."""

    agent = seeded_advanced(tmp_path, min_confidence=0.0)
    before = agent.profile_store.facts("u")
    agent.reply("u", "t", message)
    assert agent.profile_store.facts("u") != before


def test_confident_statements_and_corrections_pass_threshold() -> None:
    assert extract_profile_updates("Mình đã chuyển ra Hà Nội rồi nhé.") == {"location": "Hà Nội"}
    assert extract_profile_updates("À mình đính chính, hiện tại mình làm data engineer.") == {
        "profession": "data engineer"
    }
    # "…nếu cần" at the end is not a hypothetical; a leading "Nếu" is.
    assert extract_profile_updates("Bạn nhớ là mình đang ở Huế để dùng ví dụ địa phương nếu cần.") == {
        "location": "Huế"
    }
    assert extract_profile_updates("Nếu mình chuyển ra Hà Nội thì sẽ báo bạn.") == {}


def test_repeated_pending_fact_is_promoted(tmp_path: Path) -> None:
    agent = seeded_advanced(tmp_path)
    agent.reply("u", "t1", "Hình như món ăn yêu thích là phở.")
    assert agent.profile_store.facts("u")["food"] == "mì Quảng"
    assert ("food", "phở") in agent.pending_facts["u"]

    agent.reply("u", "t2", "Hình như món ăn yêu thích của mình là phở.")  # second, independent mention
    assert agent.profile_store.facts("u")["food"] == "phở"
    assert ("food", "phở") not in agent.pending_facts["u"]


def test_dataset_recall_unchanged_by_threshold(tmp_path: Path) -> None:
    from benchmark import run_agent_benchmark

    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name in ("conversations.json", "advanced_long_context.json"):
        config = dataclasses.replace(make_config(tmp_path / name), compact_threshold_tokens=500)
        row = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), load_conversations(data_dir / name), config)
        assert row.recall_score == 1.0


def test_parallel_upserts_do_not_lose_facts(tmp_path: Path) -> None:
    """Live mode runs tool calls in parallel; a racy read-modify-write once wiped User.md."""

    import threading

    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("u", "name", "DũngCT")
    store.upsert_fact("u", "drink", "cà phê sữa đá")
    threads = [threading.Thread(target=store.upsert_fact, args=("u", f"fact_{i}", "x" * 50)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    facts = store.facts("u")
    assert facts["name"] == "DũngCT" and facts["drink"] == "cà phê sữa đá"
    assert all(f"fact_{i}" in facts for i in range(16))
    assert not list((tmp_path / "profiles" / "u").glob("*.tmp"))  # no leftover temp files


def test_upsert_normalizes_llm_keys_and_values(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.upsert_fact("u", "Món ăn", "bún bò\nHuế")
    assert store.facts("u") == {"món_ăn": "bún bò Huế"}  # still one parseable line
    assert not store.upsert_fact("u", "  ", "x")
