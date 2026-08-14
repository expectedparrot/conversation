# Async backend evaluation

Evaluated for the 0.2.0 release against EDSL 1.0.8.

## Findings

- EDSL's native `Jobs.run_async()` completes successfully with its offline test
  model and returns the same `Results` shape used by synchronous runs.
- Conversation state, stopping checks, speaker selection, retry classification,
  and statement recording are shared with or mirror the synchronous contract.
- Async retry delays use `asyncio.sleep()` and do not block the event loop.
- `ConversationList.run_async()` bounds active conversations with a semaphore and
  propagates failures with the conversation index and original cause.
- Cancellation propagates as `CancelledError`; an interrupted turn is not appended
  to the transcript. Sibling tasks are cancelled and awaited during batch cleanup.
- The package never calls `asyncio.run()`, so it works inside an existing event loop.

The offline suite verifies all of these properties. The opt-in integration test
exercises a real Google-backed conversation when provider credentials are present.
That test is the coverage point for Coop/provider result retrieval because the
`results_uuid` and immediately-available response variants cannot be generated
faithfully without the remote service.

## API decision

The synchronous API remains the default for scripts and moderate batches. Native
`converse_async()` and `ConversationList.run_async()` are peer APIs for event-loop
applications; neither API wraps the other, and existing synchronous callers require
no migration.

Do not run the same `Conversation` instance concurrently through both APIs. A
conversation owns mutable transcript and strategy state even though its participant
agents remain caller-owned.
