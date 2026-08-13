import copy

import pytest
from conftest import make_result
from edsl import QuestionFreeText

from conversation import (
    AgentStatement,
    Conversation,
    ConversationList,
    ConversationValueError,
)


def completed_conversation(agents, monkeypatch):
    conversation = Conversation(
        agent_list=agents,
        max_turns=2,
        verbose=True,
        conversation_index=4,
        per_round_message_template="Turn {{ current_turn }}",
    )
    monkeypatch.setattr(
        conversation,
        "_get_next_statement",
        lambda *, index, speaker, conversation: make_result(
            speaker, f"answer {index}", index
        ),
    )
    conversation.converse()
    return conversation


def test_conversation_round_trip_preserves_supported_state(agents, monkeypatch):
    conversation = completed_conversation(agents, monkeypatch)

    restored = Conversation.from_dict(conversation.to_dict())

    assert restored.max_turns == 2
    assert restored.verbose is True
    assert restored.conversation_index == 4
    assert restored.per_round_message_template == "Turn {{ current_turn }}"
    assert restored.agent_statements.transcript == conversation.agent_statements.transcript
    assert restored.next_statement_question.to_dict() == (
        conversation.next_statement_question.to_dict()
    )


def test_custom_question_round_trips(agents):
    question = QuestionFreeText(
        question_name="dialogue",
        question_text="Speak as {{ speaker_name }}: {{ conversation }}",
    )
    conversation = Conversation(agent_list=agents, next_statement_question=question)

    restored = Conversation.from_dict(conversation.to_dict())

    assert restored.next_statement_question.question_text == question.question_text


def test_conversation_list_round_trips(agents, monkeypatch):
    original = ConversationList([completed_conversation(agents, monkeypatch)])

    restored = ConversationList.from_dict(original.to_dict())

    assert len(restored.conversations) == 1
    assert restored.conversations[0].agent_statements.transcript == (
        original.conversations[0].agent_statements.transcript
    )


def test_legacy_serialization_without_version_is_accepted(agents):
    data = Conversation(agent_list=agents).to_dict()
    data.pop("serialization_version")
    data.pop("next_statement_question")
    data.pop("per_round_message_template")

    restored = Conversation.from_dict(data)

    assert restored.max_turns == 20


def test_unknown_serialization_version_is_rejected(agents):
    data = Conversation(agent_list=agents).to_dict()
    data["serialization_version"] = 999

    with pytest.raises(ConversationValueError, match="Unsupported"):
        Conversation.from_dict(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stopping_function": lambda statements: False},
        {"next_speaker_generator": lambda agent_list, **kwargs: agent_list[0]},
    ],
)
def test_unsupported_callables_fail_explicitly(agents, kwargs):
    conversation = Conversation(agent_list=agents, **kwargs)

    with pytest.raises(ConversationValueError, match="cannot be serialized"):
        conversation.to_dict()


def test_serialized_data_does_not_alias_statement_data(agents, monkeypatch):
    conversation = completed_conversation(agents, monkeypatch)
    data = copy.deepcopy(conversation.to_dict())

    restored = Conversation.from_dict(data)
    data["agent_statements"].clear()

    assert len(restored.agent_statements) == 2


def test_restored_partial_conversation_resumes_turn_and_speaker(agents, monkeypatch):
    original = Conversation(agent_list=agents, max_turns=3)
    original.agent_statements.append(
        AgentStatement(make_result(agents[0], "first", 0))
    )

    restored = Conversation.from_dict(original.to_dict())
    observed = []

    def fake_turn(*, index, speaker, conversation):
        observed.append((index, speaker.name))
        return make_result(speaker, f"turn {index}", index)

    monkeypatch.setattr(restored, "_get_next_statement", fake_turn)
    restored.converse()

    assert observed == [(1, "Bob"), (2, "Alice")]
    assert len(restored.agent_statements) == 3
