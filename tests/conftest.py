import pytest
from edsl import Agent, AgentList, Model, Scenario
from edsl.results.result import Result


@pytest.fixture
def agents():
    participants = AgentList([Agent(name="Alice"), Agent(name="Bob")])
    for agent in participants:
        agent.model = Model("test")
    return participants


def make_result(agent, text, index=0):
    model = getattr(agent, "model", None) or Model("test")
    return Result(
        agent=agent,
        scenario=Scenario({"index": index}),
        model=model,
        iteration=0,
        answer={"dialogue": text},
    )
