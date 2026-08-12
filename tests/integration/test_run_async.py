"""Credential-dependent EDSL async integration smoke test."""

import pytest
from edsl import Agent, Model, QuestionFreeText, Scenario


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_async():
    question = QuestionFreeText(question_text="Say hello", question_name="greeting")
    model = Model("gemini-2.0-flash", service_name="google")
    agent = Agent(name="TestAgent", traits={"role": "friendly"})
    jobs = question.by(Scenario({"context": "test"})).by(agent).by(model)

    results = await jobs.run_async(disable_remote_inference=False)

    assert len(results) == 1
