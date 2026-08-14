"""Credential-dependent Conversation async integration smoke test."""

import pytest
from edsl import Agent, AgentList, Model

from conversation import Conversation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_async():
    model = Model("gemini-2.0-flash", service_name="google")
    agent = Agent(name="TestAgent", traits={"role": "friendly"})
    conversation = Conversation(
        agent_list=AgentList([agent]), max_turns=1, default_model=model
    )

    await conversation.converse_async()

    assert len(conversation.agent_statements) == 1
