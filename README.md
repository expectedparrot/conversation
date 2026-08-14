# conversation

A standalone package for simulating multi-agent conversations using [EDSL](https://github.com/expectedparrot/edsl) and [Expected Parrot](https://www.expectedparrot.com) (Coop) for language model inference.

## Overview

Each conversation is a sequence of turns between EDSL `Agent` objects. At every turn, the current speaker is asked a `QuestionFreeText` whose prompt contains the conversation history so far. The answer becomes the next statement, and the cycle continues until a turn limit or stopping condition is reached.

Multiple conversations can be run in parallel via `ConversationList`, which uses a
bounded thread pool. Each worker blocks independently on its own Coop jobs, so many
conversations make progress simultaneously without requiring an async event loop.

## Installation

```bash
pip install -e /path/to/conversation
```

Requires an [Expected Parrot API key](https://www.expectedparrot.com):

```bash
export EXPECTED_PARROT_API_KEY=your_key_here
```

## Quick start

```python
from edsl import Agent, AgentList
from conversation import Conversation

alice = Agent(name="Alice", traits={"motivation": "You are a curious scientist."})
bob   = Agent(name="Bob",   traits={"motivation": "You are a skeptical philosopher."})

c = Conversation(agent_list=AgentList([alice, bob]), max_turns=6, verbose=True)
c.converse()
```

Each turn is submitted to Coop as a normal EDSL job and blocks until the result returns. With `verbose=True` each statement is printed as it arrives.

## How it works

### Turn loop

`Conversation.converse()` runs a simple sequential loop:

1. Ask `_should_continue()` — stops when `max_turns` is reached or a custom `stopping_function` returns `True`.
2. Call `next_speaker()` to get the current speaker (default: round-robin).
3. Build a one-question EDSL job with `_build_job()`.
4. Call `job.run()` which submits to Coop and blocks until done.
5. Append the result to `agent_statements` and advance the turn counter.

The conversation history is passed into every job as a `Scenario` field (`conversation`), so each agent sees the full transcript before responding.

### The turn prompt

The default question template is:

```
You are {{ speaker_name }}. This is the conversation so far: {{ conversation }}
{% if round_message is not none %}
{{ round_message }}
{% endif %}
What do you say next?
```

`{{ conversation }}` is a list of `{speaker_name: text}` dicts representing the transcript so far. `{{ round_message }}` is an optional per-round injection (see `per_round_message_template`).

You can replace the entire prompt by passing a custom `QuestionFreeText` (or any `QuestionBase`) as `next_statement_question`. The question must use `question_name="dialogue"` so that `AgentStatement.text` can find the answer.

### Parallelism

`ConversationList` submits conversations to a `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(c.converse) for c in self.conversations]
    for future in as_completed(futures):
        future.result()
```

Because turns block on network I/O (waiting for Coop), the GIL is released and threads make real concurrent progress. While conversation A is waiting for turn 3, conversation B is already waiting for its turn 3 on a different Coop job. No `asyncio` event loop is needed. Worker failures propagate from `run()` with the failed conversation index and original exception attached as the cause.

### Why not asyncio?

`job.run()` is synchronous and uses `poll_remote_inference_job` internally, which correctly handles both the `results_uuid` and `results_available` response paths from Coop. The async alternative (`run_async` / `results.fetch()`) is missing the `results_available` fallback and fails silently on some accounts. Threads give equivalent parallelism for the dozens-to-hundreds of conversations this module targets, with simpler code and no event loop management.

### Retry behaviour

`converse()` retries only on typed connection/timeout failures and structured HTTP
502, 503, or 504 responses. Application-level errors propagate immediately.
Defaults: 3 total attempts and a 5 second delay between attempts. The original
exception is re-raised after exhaustion, and there is no delay after the final
attempt.

```python
c.converse(max_retries=5, retry_delay=10.0)
```

## API reference

### `Conversation`

```python
Conversation(
    agent_list,                  # AgentList of participants
    max_turns=20,                # hard cap on number of turns
    stopping_function=None,      # callable(AgentStatements) -> bool
    next_statement_question=None,# custom QuestionBase (must have question_name="dialogue")
    next_speaker_generator=None, # callable — see next_speaker_utilities
    verbose=False,               # print each statement as it arrives
    per_round_message_template=None,  # Jinja template injected each round
    conversation_index=None,     # set automatically by ConversationList
    cache=None,                  # edsl Cache object
    default_model=None,          # edsl Model applied to agents without one
)
```

`agent_list` must be a non-empty `AgentList`, and `max_turns` must be a
non-negative integer. One-agent conversations are supported by every built-in
turn-taking strategy. Custom questions must use `question_name="dialogue"`.

#### Methods

| Method | Description |
|--------|-------------|
| `converse(max_retries=3, retry_delay=5.0)` | Run the conversation to completion |
| `reset()` | Clear completed turns and restart at turn zero |
| `to_results()` | Return all statements as an EDSL `Results` object |
| `summarize()` | Return a `Scenario` with transcript and metadata, suitable for follow-up analysis |
| `to_dict()` / `from_dict()` | Serialization |

Calling `converse()` again continues from the existing transcript until `max_turns`
is reached; calling it after completion is a no-op. Restored conversations resume
with the next stable turn index and, for the serializable default strategy, the same
speaker order as uninterrupted execution. Call `reset()` to start over explicitly.

Serialization uses a versioned dictionary format and preserves agents, accumulated
statements, question configuration, round-message templates, and conversation
metadata. Custom stopping functions and speaker-generator callables are rejected
explicitly because arbitrary Python callables cannot be restored faithfully. Cache
objects are runtime-only and are not serialized.

#### `stopping_function`

Called after each turn with the current `AgentStatements`. Return `True` to end the conversation early:

```python
def stop_on_deal(statements):
    if statements:
        return "DEAL" in statements[-1].text.upper()
    return False

c = Conversation(agent_list=..., stopping_function=stop_on_deal)
```

#### `per_round_message_template`

A Jinja template rendered each turn and injected as `{{ round_message }}` into the question. Available variables: `max_turns`, `current_turn`.

```python
c = Conversation(
    agent_list=...,
    per_round_message_template="You are on turn {{ current_turn }} of {{ max_turns }}. Start wrapping up.",
)
```

Your `next_statement_question` must include `{{ round_message }}` in its text when this is set.

#### `next_statement_question`

Replace the default prompt entirely:

```python
from edsl import QuestionFreeText

q = QuestionFreeText(
    question_text="""
You are {{ speaker_name }}. Conversation so far: {{ conversation }}
Respond in character, in under 50 words.
""",
    question_name="dialogue",
)
c = Conversation(agent_list=..., next_statement_question=q)
```

### `ConversationList`

```python
ConversationList(conversations)   # list of Conversation objects
```

An empty `ConversationList` is valid and produces empty results and summaries.
Pass a positive integer to `run(max_workers=...)` to bound concurrency.

| Method | Description |
|--------|-------------|
| `run(max_workers=None)` | Run conversations concurrently, optionally bounding the worker count |
| `to_results()` | Concatenate results from all conversations into one `Results` |
| `summarize()` | Return a `ScenarioList` of per-conversation summaries |
| `to_dict()` / `from_dict()` | Serialization |

### Turn-taking generators

Imported from `conversation.next_speaker_utilities`:

| Generator | Behaviour |
|-----------|-----------|
| `default_turn_taking_generator` | Round-robin (default) |
| `turn_taking_generator_with_focal_speaker` | One agent always speaks between others (e.g. an auctioneer) |
| `random_turn_taking_generator` | Random, no consecutive repeats |
| `random_inclusive_generator` | Random, every agent speaks once before anyone repeats |

Pass via `next_speaker_generator`:

```python
from conversation.next_speaker_utilities import random_inclusive_generator

c = Conversation(agent_list=..., next_speaker_generator=random_inclusive_generator)
```

For configurable, reproducible, and serializable turn taking, prefer the explicit
strategy classes:

```python
from conversation import FocalSpeakerStrategy, RandomSpeakerStrategy

focal = FocalSpeakerStrategy(focal_speaker_index=0)
seeded_random = RandomSpeakerStrategy(seed=42)

c = Conversation(agent_list=..., next_speaker_generator=seeded_random)
```

Strategies expose their `speakers_so_far` state and support `reset(history)`. The
legacy generator functions and arbitrary callables remain supported, but arbitrary
callables cannot be serialized.

### Working with results

After a conversation runs, `to_results()` returns a standard EDSL `Results` object:

```python
results = c.to_results()
results.select("agent.agent_name", "answer.dialogue").print(format="rich")
```

Use `summarize()` to get a `Scenario` containing the full transcript, suitable for passing to follow-up EDSL survey questions:

```python
from edsl import QuestionYesNo, QuestionFreeText

summary = c.summarize()   # Scenario with keys: transcript, num_agents, max_turns, ...

q_deal = QuestionYesNo(
    question_text="Transcript: {{ transcript }}. Was a deal reached?",
    question_name="deal_reached",
)
q_price = QuestionFreeText(
    question_text="Transcript: {{ transcript }}. What was the final offer from each side?",
    question_name="price_summary",
)

analysis = q_deal.add_question(q_price).by(summary).run()
print(analysis.select("answer.deal_reached").to_list()[0])
print(analysis.select("answer.price_summary").to_list()[0])
```

For `ConversationList`, use `summarize()` to get a `ScenarioList` and run the same analysis across all conversations at once:

```python
analysis = q_deal.add_question(q_price).by(cl.summarize()).run()
```

## Examples

See the `examples/` directory:

- `car_buying.py` — three-agent conversation (buyer, salesman, skeptical brother-in-law) run in parallel
- `mug_negotiation.py` — bilateral bargaining across multiple valuation pairs with post-hoc deal analysis
- `chips.py` — chip-trading negotiation using a custom `Agent` subclass with internal state

## Development

Install the package with its test dependencies and run the offline suite:

```bash
python -m pip install -e ".[test]"
python -m ruff check .
python -m pytest
```

Tests marked `integration` require external credentials and are excluded by default.
Run them explicitly with `python -m pytest -m integration` after configuring the
required model-provider credentials.

## Package structure

```
conversation/
├── Conversation.py          # Conversation, ConversationList, AgentStatement, AgentStatements
├── exceptions.py            # ConversationError, ConversationValueError, ConversationStateError
├── next_speaker_utilities.py# Turn-taking generator functions
└── __init__.py
examples/
├── car_buying.py
├── mug_negotiation.py
└── chips.py
tests/
└── test_run_async.py
pyproject.toml
```
