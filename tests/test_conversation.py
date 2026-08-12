import importlib

import pytest
import requests
from conftest import make_result

from conversation import Conversation


def test_converse_runs_to_max_turns_in_round_robin_order(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=3)

    def fake_turn(*, index, speaker, conversation):
        assert len(conversation) == index
        return make_result(speaker, f"turn {index}", index)

    monkeypatch.setattr(conversation, "_get_next_statement", fake_turn)
    conversation.converse()

    assert conversation.agent_statements.transcript == [
        {"Alice": "turn 0"},
        {"Bob": "turn 1"},
        {"Alice": "turn 2"},
    ]


def test_stopping_function_can_stop_after_a_turn(agents, monkeypatch):
    conversation = Conversation(
        agent_list=agents,
        max_turns=10,
        stopping_function=lambda statements: len(statements) == 1,
    )
    monkeypatch.setattr(
        conversation,
        "_get_next_statement",
        lambda *, index, speaker, conversation: make_result(speaker, "done", index),
    )

    conversation.converse()

    assert len(conversation.agent_statements) == 1


def test_stopping_function_can_prevent_first_turn(agents, monkeypatch):
    conversation = Conversation(
        agent_list=agents,
        stopping_function=lambda statements: True,
    )
    fetch = monkeypatch.setattr(
        conversation,
        "_get_next_statement",
        lambda **kwargs: pytest.fail("no turn should be fetched"),
    )

    conversation.converse()

    assert fetch is None
    assert conversation.agent_statements == []


def test_transient_error_is_retried(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0

    def flaky_turn(*, index, speaker, conversation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary connection failure")
        return make_result(speaker, "recovered", index)

    monkeypatch.setattr(conversation, "_get_next_statement", flaky_turn)
    conversation_module = importlib.import_module("conversation.Conversation")
    monkeypatch.setattr(conversation_module.time, "sleep", lambda delay: None)

    conversation.converse(max_retries=2)

    assert attempts == 2
    assert conversation.agent_statements[0].text == "recovered"


def test_non_transient_error_is_not_retried(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0

    def broken_turn(**kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid response")

    monkeypatch.setattr(conversation, "_get_next_statement", broken_turn)

    with pytest.raises(ValueError, match="invalid response"):
        conversation.converse(max_retries=3, retry_delay=0)

    assert attempts == 1


def test_transient_error_is_preserved_after_retry_exhaustion(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0

    def broken_turn(**kwargs):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("timeout")

    monkeypatch.setattr(conversation, "_get_next_statement", broken_turn)
    conversation_module = importlib.import_module("conversation.Conversation")
    delays = []
    monkeypatch.setattr(conversation_module.time, "sleep", delays.append)

    with pytest.raises(TimeoutError, match="timeout") as exc_info:
        conversation.converse(max_retries=2)

    assert attempts == 2
    assert delays == [5.0]
    assert any("turn 0" in note for note in getattr(exc_info.value, "__notes__", []))


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_structured_server_status_is_retried(agents, monkeypatch, status_code):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0

    def flaky_turn(*, index, speaker, conversation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = requests.Response()
            response.status_code = status_code
            raise requests.HTTPError("server unavailable", response=response)
        return make_result(speaker, "recovered", index)

    monkeypatch.setattr(conversation, "_get_next_statement", flaky_turn)
    conversation_module = importlib.import_module("conversation.Conversation")
    monkeypatch.setattr(conversation_module.time, "sleep", lambda delay: None)

    conversation.converse(max_retries=2)

    assert attempts == 2


def test_error_message_does_not_make_error_retryable(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0

    def broken_turn(**kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("connection timeout with status 503")

    monkeypatch.setattr(conversation, "_get_next_statement", broken_turn)

    with pytest.raises(ValueError, match="connection timeout"):
        conversation.converse(max_retries=3, retry_delay=0)

    assert attempts == 1


def test_to_results_and_summary_use_completed_statements(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=2, conversation_index=7)
    monkeypatch.setattr(
        conversation,
        "_get_next_statement",
        lambda *, index, speaker, conversation: make_result(
            speaker, f"answer {index}", index
        ),
    )
    conversation.converse()

    results = conversation.to_results()
    summary = conversation.summarize()

    assert len(results) == 2
    assert summary["conversation_index"] == 7
    assert summary["number_of_agent_statements"] == 2
    assert summary["transcript"] == [("Alice", "answer 0"), ("Bob", "answer 1")]


def test_round_message_is_rendered_into_job_scenario(agents):
    conversation = Conversation(
        agent_list=agents,
        max_turns=4,
        per_round_message_template="Turn {{ current_turn }} of {{ max_turns }}",
    )

    job = conversation._build_job(index=2, speaker=agents[0], conversation=[])

    assert job.scenarios[0]["round_message"] == "Turn 2 of 4"


def test_custom_question_requires_round_message_placeholder(agents):
    from edsl import QuestionFreeText

    from conversation import ConversationValueError

    question = QuestionFreeText(
        question_name="dialogue",
        question_text="Conversation: {{ conversation }}",
    )

    with pytest.raises(ConversationValueError):
        Conversation(
            agent_list=agents,
            next_statement_question=question,
            per_round_message_template="Wrap up",
        )
