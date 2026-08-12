import random

from conversation.next_speaker_utilities import (
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
