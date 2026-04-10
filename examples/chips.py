from typing import Optional

from edsl import Agent, AgentList, Model
from conversation import Conversation, ConversationList


class ChipLover(Agent):
    def __init__(self, name, chip_values, initial_chips, model: Optional[Model] = None):
        self.chip_values = chip_values
        self.initial_chips = initial_chips
        self.current_chips = initial_chips
        self.model = model or Model()
        super().__init__(
            name=name,
            traits={
                "motivation": f"""
            You are {name}. You are negotiating the trading of colored 'chips' with other players. You want to maximize your score.
            When you want to accept a deal, say "DEAL!"
            Note that different players can have different values for the chips.
            """,
                "chip_values": chip_values,
                "initial_chips": initial_chips,
            },
        )

    def trade(self, chips_given_dict, chips_received_dict):
        for color, amount in chips_given_dict.items():
            self.current_chips[color] -= amount
        for color, amount in chips_received_dict.items():
            self.current_chips[color] += amount

    def get_score(self):
        return sum(
            self.chip_values[color] * self.current_chips[color]
            for color in self.chip_values
        )


a1 = ChipLover(
    name="Alice",
    chip_values={"Green": 7, "Blue": 1, "Red": 0},
    model=Model("gemini-2.0-flash"),
    initial_chips={"Green": 1, "Blue": 2, "Red": 3},
)
a2 = ChipLover(
    name="Bob",
    chip_values={"Green": 7, "Blue": 1, "Red": 0},
    model=Model("gemini-2.0-flash"),
    initial_chips={"Green": 1, "Blue": 2, "Red": 3},
)

c1 = Conversation(agent_list=AgentList([a1, a2]), max_turns=2, verbose=True)

combo = ConversationList([c1])
combo.run()
