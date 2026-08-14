"""
The conversation module provides tools for simulating conversations between agents.

It includes classes for managing dialogues, tracking statements, and controlling
conversation flow between multiple participants.
"""

from .Conversation import (
    AgentStatement,
    AgentStatements,
    Conversation,
    ConversationList,
)
from .exceptions import (
    ConversationError,
    ConversationStateError,
    ConversationValueError,
)
from .next_speaker_utilities import (
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

__all__ = [
    "Conversation",
    "ConversationList",
    "AgentStatement",
    "AgentStatements",
    "ConversationError",
    "ConversationValueError",
    "ConversationStateError",
    "default_turn_taking_generator",
    "turn_taking_generator_with_focal_speaker",
    "random_turn_taking_generator",
    "random_inclusive_generator",
    "speaker_closure",
    "SpeakerStrategy",
    "RoundRobinStrategy",
    "FocalSpeakerStrategy",
    "RandomSpeakerStrategy",
    "RandomInclusiveStrategy",
]
