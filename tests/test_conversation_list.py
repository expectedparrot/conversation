import threading
import time

import pytest
from conftest import make_result

from conversation import Conversation, ConversationList


def test_run_executes_conversations_and_aggregates_results(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(2)]
    thread_ids = set()

    for conversation in conversations:
        def fake_turn(*, index, speaker, conversation, _ids=thread_ids):
            _ids.add(threading.get_ident())
            return make_result(speaker, "hello", index)

        monkeypatch.setattr(conversation, "_get_next_statement", fake_turn)

    conversation_list = ConversationList(conversations)
    conversation_list.run()

    assert [c.conversation_index for c in conversations] == [0, 1]
    assert len(conversation_list.to_results()) == 2
    assert all(len(c.agent_statements) == 1 for c in conversations)
    assert thread_ids


def test_summarize_returns_one_scenario_per_conversation(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(2)]
    for conversation in conversations:
        monkeypatch.setattr(
            conversation,
            "_get_next_statement",
            lambda *, index, speaker, conversation: make_result(speaker, "hello", index),
        )
        conversation.converse()
    conversation_list = ConversationList(conversations)

    summaries = conversation_list.summarize()

    assert len(summaries) == 2
    assert [summary["conversation_index"] for summary in summaries] == [0, 1]


def test_run_propagates_worker_failure_with_conversation_index(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(2)]
    monkeypatch.setattr(
        conversations[0],
        "_get_next_statement",
        lambda **kwargs: make_result(kwargs["speaker"], "hello", kwargs["index"]),
    )
    monkeypatch.setattr(
        conversations[1],
        "_get_next_statement",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad response")),
    )

    with pytest.raises(RuntimeError, match="Conversation 1 failed") as exc_info:
        ConversationList(conversations).run()

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "bad response"


def test_run_respects_max_workers(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(3)]
    active = 0
    highest_active = 0
    lock = threading.Lock()

    def slow_turn(*, index, speaker, conversation):
        nonlocal active, highest_active
        with lock:
            active += 1
            highest_active = max(highest_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return make_result(speaker, "hello", index)

    for conversation in conversations:
        monkeypatch.setattr(conversation, "_get_next_statement", slow_turn)

    ConversationList(conversations).run(max_workers=1)

    assert highest_active == 1


def test_run_accepts_an_empty_conversation_list():
    ConversationList([]).run(max_workers=1)
