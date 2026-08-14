import random
from abc import ABC, abstractmethod

from .exceptions import ConversationValueError


def _validate_agents(agent_list):
    if len(agent_list) == 0:
        raise ConversationValueError("agent_list must contain at least one agent")


class SpeakerStrategy(ABC):
    """Explicit, resettable state for selecting conversation speakers."""

    strategy_name = None

    def __init__(self):
        self.speakers_so_far = []

    def reset(self, speakers_so_far=None):
        self.speakers_so_far = list(speakers_so_far or [])

    def next_speaker(self, agent_list):
        _validate_agents(agent_list)
        speaker = self._select(agent_list)
        self.speakers_so_far.append(speaker)
        return speaker

    @abstractmethod
    def _select(self, agent_list):
        pass

    def to_dict(self):
        if self.strategy_name is None:
            raise ConversationValueError(
                "Custom speaker strategies cannot be serialized"
            )
        return {"strategy_name": self.strategy_name}

    @classmethod
    def from_dict(cls, data):
        strategy_types = {
            "round_robin": RoundRobinStrategy,
            "focal": FocalSpeakerStrategy,
            "random_no_repeat": RandomSpeakerStrategy,
            "random_inclusive": RandomInclusiveStrategy,
        }
        strategy_name = data.get("strategy_name")
        if strategy_name not in strategy_types:
            raise ConversationValueError(
                f"Unknown speaker strategy: {strategy_name!r}"
            )
        kwargs = {key: value for key, value in data.items() if key != "strategy_name"}
        return strategy_types[strategy_name](**kwargs)


class RoundRobinStrategy(SpeakerStrategy):
    strategy_name = "round_robin"

    def _select(self, agent_list):
        return default_turn_taking_generator(agent_list, self.speakers_so_far)


class FocalSpeakerStrategy(SpeakerStrategy):
    strategy_name = "focal"

    def __init__(self, focal_speaker_index=0):
        super().__init__()
        self.focal_speaker_index = focal_speaker_index

    def _select(self, agent_list):
        return turn_taking_generator_with_focal_speaker(
            agent_list, self.speakers_so_far, self.focal_speaker_index
        )

    def to_dict(self):
        return {
            **super().to_dict(),
            "focal_speaker_index": self.focal_speaker_index,
        }


class _SeededStrategy(SpeakerStrategy):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, speakers_so_far=None):
        history = list(speakers_so_far or [])
        self.speakers_so_far = []
        self._rng = random.Random(self.seed)
        # Advance deterministically to the point immediately after the history.
        self._history_to_replay = history

    def replay(self, agent_list):
        history = getattr(self, "_history_to_replay", [])
        self._history_to_replay = []
        for expected in history:
            actual = super().next_speaker(agent_list)
            if actual != expected:
                raise ConversationValueError(
                    "Speaker history does not match the seeded strategy"
                )

    def next_speaker(self, agent_list):
        if getattr(self, "_history_to_replay", []):
            self.replay(agent_list)
        return super().next_speaker(agent_list)

    def to_dict(self):
        return {**super().to_dict(), "seed": self.seed}


class RandomSpeakerStrategy(_SeededStrategy):
    strategy_name = "random_no_repeat"

    def _select(self, agent_list):
        if len(agent_list) == 1:
            return agent_list[0]
        if not self.speakers_so_far:
            return self._rng.choice(agent_list)
        return self._rng.choice(
            [agent for agent in agent_list if agent != self.speakers_so_far[-1]]
        )


class RandomInclusiveStrategy(_SeededStrategy):
    strategy_name = "random_inclusive"

    def _select(self, agent_list):
        if len(agent_list) == 1:
            return agent_list[0]
        lookback_length = len(self.speakers_so_far) % len(agent_list)
        eligible = (
            list(agent_list)
            if lookback_length == 0
            else [
                agent
                for agent in agent_list
                if agent not in self.speakers_so_far[-lookback_length:]
            ]
        )
        return self._rng.choice(eligible)


class CallableSpeakerStrategy(SpeakerStrategy):
    def __init__(self, generator_function):
        super().__init__()
        if not callable(generator_function):
            raise ConversationValueError("generator_function must be callable")
        self.generator_function = generator_function

    def _select(self, agent_list):
        return self.generator_function(
            agent_list=agent_list,
            speakers_so_far=self.speakers_so_far,
            focal_speaker_index=None,
        )


def default_turn_taking_generator(agent_list, speakers_so_far, **kwargs):
    """Returns the next speaker in the list of agents, in order. If no speakers have spoken yet, returns the first agent in the list."""
    _validate_agents(agent_list)
    if len(speakers_so_far) == 0:
        return agent_list[0]
    most_recent_speaker = speakers_so_far[-1]
    index = agent_list.index(most_recent_speaker)
    return agent_list[(index + 1) % len(agent_list)]


def turn_taking_generator_with_focal_speaker(
    agent_list, speakers_so_far, focal_speaker_index, **kwargs
):
    """Returns the focal speaker first, then the next speaker in the list of agents sequentially, going back to the focal speaker after the last agent has spoken. If no speakers have spoken yet, returns the focal speaker.
    This would be appropriate for say, a an auction where the auctioneer always speaks first, and then the next person in line speaks, and then the auctioneer speaks again.
    """
    _validate_agents(agent_list)
    if (
        not isinstance(focal_speaker_index, int)
        or isinstance(focal_speaker_index, bool)
        or not 0 <= focal_speaker_index < len(agent_list)
    ):
        raise ConversationValueError(
            "focal_speaker_index must identify an agent in agent_list"
        )
    if len(agent_list) == 1 or len(speakers_so_far) == 0:
        return agent_list[focal_speaker_index]

    most_recent_speaker = speakers_so_far[-1]
    if most_recent_speaker == agent_list[focal_speaker_index]:
        non_focal_agents = [
            a for a in agent_list if a != agent_list[focal_speaker_index]
        ]
        non_focal_speakers = [
            a for a in speakers_so_far if a != agent_list[focal_speaker_index]
        ]
        return default_turn_taking_generator(
            agent_list=non_focal_agents, speakers_so_far=non_focal_speakers
        )
    else:
        return agent_list[focal_speaker_index]


def random_turn_taking_generator(agent_list, speakers_so_far, **kwargs):
    """Returns a random speaker from the list of agents, but ensuring no agent speaks twice in a row.
    If no speakers have spoken yet, returns a random agent."""
    _validate_agents(agent_list)
    if len(agent_list) == 1:
        return agent_list[0]
    if len(speakers_so_far) == 0:
        return random.choice(agent_list)
    else:
        most_recent_speaker = speakers_so_far[-1]
    return random.choice([a for a in agent_list if a != most_recent_speaker])


def random_inclusive_generator(agent_list, speakers_so_far, **kwargs):
    """Returns a random speaker from the list of agents, but ensuring no agent speaks twice in a row and that
    every agent has spoken before the same agent speaks again.
    """
    _validate_agents(agent_list)
    if len(agent_list) == 1:
        return agent_list[0]
    if len(speakers_so_far) > 0:
        most_recent_speaker = speakers_so_far[-1]
    else:
        most_recent_speaker = None

    lookback_length = len(speakers_so_far) % len(agent_list)

    if lookback_length == 0:
        eligible_agents = agent_list[:]
    else:
        eligible_agents = [
            a for a in agent_list if a not in speakers_so_far[-lookback_length:]
        ]

    # don't have the same speaker twice in a row
    if most_recent_speaker in eligible_agents:
        eligible_agents.pop(eligible_agents.index(most_recent_speaker))

    return random.choice(eligible_agents)


def speaker_closure(
    agent_list, generator_function, focal_speaker_index=None, speakers_so_far=None
):
    _validate_agents(agent_list)
    if not callable(generator_function):
        raise ConversationValueError("generator_function must be callable")
    speakers_so_far = list(speakers_so_far or [])
    focal_speaker_index = focal_speaker_index

    def next_speaker_generator():
        speaker = generator_function(
            agent_list=agent_list,
            speakers_so_far=speakers_so_far,
            focal_speaker_index=focal_speaker_index,
        )
        speakers_so_far.append(speaker)
        return speaker

    return next_speaker_generator
