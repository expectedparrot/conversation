# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## 0.2.0 - Unreleased

### Added

- Offline behavioral tests and multi-version CI.
- Explicit, serializable speaker strategies with deterministic random seeds.
- Resumable conversations and explicit reset semantics.
- Stable identity-bearing transcript records.
- Native async conversation and bounded batch APIs.

### Changed

- Conversation batches use bounded structured thread execution.
- Retry classification uses typed exceptions and structured HTTP statuses.
- Agents remain caller-owned; model bindings are stored per conversation.
- Runtime and constructor inputs fail early with package-specific errors.

### Fixed

- Worker exceptions now propagate to callers.
- Conversation state and statements survive serialization round trips.
- Empty collections and single-agent strategies have defined behavior.

### Migration notes

- Prompt and summary transcript entries now contain `turn`, `speaker`,
  `speaker_index`, and `text`. The old shape is available from
  `AgentStatements.legacy_transcript`.
- Duplicate agent names are rejected.
- Exhausted retries re-raise the original exception instead of wrapping it.
