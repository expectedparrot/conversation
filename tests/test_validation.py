import pytest
from edsl import Agent, AgentList, QuestionFreeText

from conversation import Conversation, ConversationList, ConversationValueError
from conversation.next_speaker_utilities import (
    default_turn_taking_generator,
    random_inclusive_generator,
    random_turn_taking_generator,
    turn_taking_generator_with_focal_speaker,
)


@pytest.mark.parametrize("max_turns", [-1, 1.5, True, "2"])
def test_invalid_max_turns_is_rejected(agents, max_turns):
    with pytest.raises(ConversationValueError, match="max_turns"):
        Conversation(agent_list=agents, max_turns=max_turns)


def test_invalid_agent_list_is_rejected():
    with pytest.raises(ConversationValueError, match="at least one agent"):
        Conversation(agent_list=AgentList([]))
    with pytest.raises(ConversationValueError, match="AgentList"):
        Conversation(agent_list=[Agent(name="Alice")])


def test_duplicate_agent_names_are_rejected():
    with pytest.raises(ConversationValueError, match="unique"):
        Conversation(
            agent_list=AgentList([Agent(name="Alice"), Agent(name="Alice")])
        )


@pytest.mark.parametrize("max_retries", [0, -1, 1.5, True])
def test_invalid_max_retries_is_rejected(agents, max_retries):
    conversation = Conversation(agent_list=agents, max_turns=0)
    with pytest.raises(ConversationValueError, match="max_retries"):
        conversation.converse(max_retries=max_retries)


@pytest.mark.parametrize("retry_delay", [-1, "1", True])
def test_invalid_retry_delay_is_rejected(agents, retry_delay):
    conversation = Conversation(agent_list=agents, max_turns=0)
    with pytest.raises(ConversationValueError, match="retry_delay"):
        conversation.converse(retry_delay=retry_delay)


def test_custom_question_must_be_named_dialogue(agents):
    question = QuestionFreeText(
        question_name="response", question_text="Conversation: {{ conversation }}"
    )
    with pytest.raises(ConversationValueError, match="question_name='dialogue'"):
        Conversation(agent_list=agents, next_statement_question=question)


def test_empty_conversation_collections_are_supported(agents):
    summary = Conversation(agent_list=agents, max_turns=0).summarize()
    conversation_list = ConversationList([])
    assert summary["transcript"] == []
    assert len(conversation_list.to_results()) == 0
    assert len(conversation_list.summarize()) == 0


@pytest.mark.parametrize("max_workers", [0, -1, 1.5, True])
def test_invalid_max_workers_is_rejected(max_workers):
    with pytest.raises(ConversationValueError, match="max_workers"):
        ConversationList([]).run(max_workers=max_workers)


def test_conversation_list_validates_its_contents(agents):
    with pytest.raises(ConversationValueError, match="must be a list"):
        ConversationList(tuple())
    with pytest.raises(ConversationValueError, match="Conversation objects"):
        ConversationList([agents[0]])


@pytest.mark.parametrize(
    "generator", [random_turn_taking_generator, random_inclusive_generator]
)
def test_random_generators_support_one_agent(generator):
    only_agent = Agent(name="Solo")
    assert generator(AgentList([only_agent]), [only_agent]) == only_agent


@pytest.mark.parametrize(
    "generator",
    [default_turn_taking_generator, random_turn_taking_generator, random_inclusive_generator],
)
def test_generators_reject_empty_agent_list(generator):
    with pytest.raises(ConversationValueError, match="at least one agent"):
        generator(AgentList([]), [])


@pytest.mark.parametrize("index", [-1, 2, None, True])
def test_focal_generator_rejects_invalid_index(agents, index):
    with pytest.raises(ConversationValueError, match="focal_speaker_index"):
        turn_taking_generator_with_focal_speaker(agents, [], index)


def test_conversation_does_not_attach_model_to_caller_agent():
    agent = Agent(name="Alice")
    assert not hasattr(agent, "model")

    conversation = Conversation(agent_list=AgentList([agent]))

    assert not hasattr(agent, "model")
    assert conversation.model_for(agent) is not None


def test_default_model_is_bound_without_mutating_agents():
    from edsl import Model

    agent = Agent(name="Alice")
    model = Model("test")
    conversation = Conversation(agent_list=AgentList([agent]), default_model=model)

    assert conversation.model_for(agent) is model
    assert not hasattr(agent, "model")


def test_explicit_agent_model_takes_precedence():
    from edsl import Model

    agent = Agent(name="Alice")
    agent.model = Model("test", canned_response="agent")
    default = Model("test", canned_response="default")
    conversation = Conversation(
        agent_list=AgentList([agent]), default_model=default
    )

    assert conversation.model_for(agent) is agent.model


def test_model_for_rejects_non_participant(agents):
    conversation = Conversation(agent_list=agents)

    with pytest.raises(ConversationValueError, match="does not belong"):
        conversation.model_for(Agent(name="Outsider"))
