import asyncio

import pytest
from conftest import make_result

from conversation import Conversation, ConversationList, ConversationValueError


@pytest.mark.asyncio
async def test_converse_async_uses_native_async_turns(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=2)
    observed = []

    async def fake_turn(*, index, speaker, conversation):
        observed.append((index, speaker.name))
        await asyncio.sleep(0)
        return make_result(speaker, f"turn {index}", index)

    monkeypatch.setattr(conversation, "_get_next_statement_async", fake_turn)

    await conversation.converse_async()

    assert observed == [(0, "Alice"), (1, "Bob")]
    assert len(conversation.agent_statements) == 2


@pytest.mark.asyncio
async def test_converse_async_retries_without_blocking(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    attempts = 0
    delays = []

    async def flaky_turn(*, index, speaker, conversation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timeout")
        return make_result(speaker, "recovered", index)

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(conversation, "_get_next_statement_async", flaky_turn)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await conversation.converse_async(max_retries=2, retry_delay=0.25)

    assert attempts == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_conversation_list_async_bounds_concurrency(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(3)]
    active = 0
    highest_active = 0

    async def slow_turn(*, index, speaker, conversation):
        nonlocal active, highest_active
        active += 1
        highest_active = max(highest_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return make_result(speaker, "hello", index)

    for conversation in conversations:
        monkeypatch.setattr(conversation, "_get_next_statement_async", slow_turn)

    await ConversationList(conversations).run_async(max_concurrency=1)

    assert highest_active == 1


@pytest.mark.asyncio
async def test_conversation_list_async_propagates_indexed_failure(agents, monkeypatch):
    conversations = [Conversation(agent_list=agents, max_turns=1) for _ in range(2)]

    async def successful_turn(*, index, speaker, conversation):
        return make_result(speaker, "hello", index)

    async def failed_turn(**kwargs):
        raise ValueError("bad response")

    monkeypatch.setattr(conversations[0], "_get_next_statement_async", successful_turn)
    monkeypatch.setattr(conversations[1], "_get_next_statement_async", failed_turn)

    with pytest.raises(RuntimeError, match="Conversation 1 failed") as exc_info:
        await ConversationList(conversations).run_async()

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_async_cancellation_does_not_record_a_turn(agents, monkeypatch):
    conversation = Conversation(agent_list=agents, max_turns=1)
    started = asyncio.Event()

    async def blocked_turn(**kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(conversation, "_get_next_statement_async", blocked_turn)
    task = asyncio.create_task(conversation.converse_async())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert conversation.agent_statements == []


@pytest.mark.asyncio
async def test_async_rejects_invalid_concurrency():
    with pytest.raises(ConversationValueError, match="max_concurrency"):
        await ConversationList([]).run_async(max_concurrency=0)


@pytest.mark.asyncio
async def test_sync_and_async_have_equivalent_state(agents, monkeypatch):
    sync_conversation = Conversation(agent_list=agents, max_turns=2)
    async_conversation = Conversation(agent_list=agents, max_turns=2)

    def sync_turn(*, index, speaker, conversation):
        return make_result(speaker, f"turn {index}", index)

    async def async_turn(*, index, speaker, conversation):
        return make_result(speaker, f"turn {index}", index)

    monkeypatch.setattr(sync_conversation, "_get_next_statement", sync_turn)
    monkeypatch.setattr(
        async_conversation, "_get_next_statement_async", async_turn
    )

    sync_conversation.converse()
    await async_conversation.converse_async()

    assert async_conversation.transcript == sync_conversation.transcript
