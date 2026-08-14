import random

import pytest

from conversation.next_speaker_utilities import (
    FocalSpeakerStrategy,
    RandomInclusiveStrategy,
    RandomSpeakerStrategy,
    RoundRobinStrategy,
    SpeakerStrategy,
    default_turn_taking_generator,
    random_inclusive_generator,
    random_turn_taking_generator,
    speaker_closure,
    turn_taking_generator_with_focal_speaker,
)


def test_default_turn_taking_cycles_in_order(agents):
    next_speaker = speaker_closure(agents, default_turn_taking_generator)

    assert [next_speaker().name for _ in range(5)] == [
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Alice",
    ]


def test_focal_speaker_alternates_with_other_speakers(agents):
    speakers = []
    for _ in range(4):
        speaker = turn_taking_generator_with_focal_speaker(
            agents, speakers, focal_speaker_index=0
        )
        speakers.append(speaker)

    assert [speaker.name for speaker in speakers] == ["Alice", "Bob", "Alice", "Bob"]


def test_random_turn_taking_never_repeats_immediately(agents, monkeypatch):
    monkeypatch.setattr(random, "choice", lambda eligible: eligible[0])
    speakers = [agents[0]]

    next_speaker = random_turn_taking_generator(agents, speakers)

    assert next_speaker == agents[1]


def test_random_inclusive_uses_every_agent_before_repeating(agents, monkeypatch):
    monkeypatch.setattr(random, "choice", lambda eligible: eligible[0])
    speakers = []

    for _ in range(2):
        speakers.append(random_inclusive_generator(agents, speakers))

    assert set(speakers) == set(agents)


def test_round_robin_strategy_exposes_and_resets_history(agents):
    strategy = RoundRobinStrategy()

    assert strategy.next_speaker(agents).name == "Alice"
    assert strategy.next_speaker(agents).name == "Bob"
    assert [agent.name for agent in strategy.speakers_so_far] == ["Alice", "Bob"]

    strategy.reset([agents[0]])

    assert strategy.next_speaker(agents).name == "Bob"


def test_focal_strategy_is_configurable_and_serializable(agents):
    strategy = FocalSpeakerStrategy(focal_speaker_index=1)

    names = [strategy.next_speaker(agents).name for _ in range(4)]
    restored = SpeakerStrategy.from_dict(strategy.to_dict())

    assert names == ["Bob", "Alice", "Bob", "Alice"]
    assert isinstance(restored, FocalSpeakerStrategy)
    assert restored.focal_speaker_index == 1


@pytest.mark.parametrize(
    "strategy_type", [RandomSpeakerStrategy, RandomInclusiveStrategy]
)
def test_seeded_strategy_is_reproducible(strategy_type, agents):
    first = strategy_type(seed=42)
    second = strategy_type(seed=42)

    first_names = [first.next_speaker(agents).name for _ in range(8)]
    second_names = [second.next_speaker(agents).name for _ in range(8)]

    assert first_names == second_names


def test_seeded_strategy_can_resume_from_history(agents):
    uninterrupted = RandomSpeakerStrategy(seed=7)
    full_sequence = [uninterrupted.next_speaker(agents) for _ in range(5)]
    resumed = RandomSpeakerStrategy(seed=7)

    resumed.reset(full_sequence[:3])

    assert [resumed.next_speaker(agents) for _ in range(2)] == full_sequence[3:]
