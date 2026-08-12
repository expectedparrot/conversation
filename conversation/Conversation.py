import time
from collections import UserList
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

from edsl import AgentList, QuestionFreeText, Results, Scenario, ScenarioList
from edsl.questions import QuestionBase
from edsl.results.result import Result
from jinja2 import Template
from requests import exceptions as requests_exceptions

if TYPE_CHECKING:
    from edsl import Model

from .exceptions import ConversationValueError
from .next_speaker_utilities import (
    default_turn_taking_generator,
    speaker_closure,
)


def _is_non_negative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})
_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests_exceptions.ConnectionError,
    requests_exceptions.Timeout,
)


def _status_code_from_exception(exc: Exception) -> Optional[int]:
    """Return a structured HTTP status carried by an exception, if present."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_transient_error(exc: Exception) -> bool:
    return isinstance(exc, _TRANSIENT_EXCEPTIONS) or (
        _status_code_from_exception(exc) in _TRANSIENT_HTTP_STATUSES
    )


class AgentStatement:
    def __init__(self, statement: Result):
        self.statement = statement

    @property
    def agent_name(self):
        return self.statement["agent"]["name"]

    def to_dict(self):
        return self.statement.to_dict()

    @classmethod
    def from_dict(cls, data):
        return cls(Result.from_dict(data))

    @property
    def text(self):
        return self.statement["answer"]["dialogue"]


class AgentStatements(UserList):
    def __init__(self, data=None):
        super().__init__(data)

    @property
    def transcript(self):
        return [{s.agent_name: s.text} for s in self.data]

    def to_dict(self):
        return [d.to_dict() for d in self.data]

    @classmethod
    def from_dict(cls, data):
        return cls([AgentStatement.from_dict(d) for d in data])


class Conversation:
    """A conversation between a list of agents.

    Each turn is run as a normal edsl job (background=False) or dispatched to
    Coop and polled until complete (background=True).  ConversationList runs
    multiple conversations in parallel using threads, so no local event loop
    is required.
    """

    def __init__(
        self,
        agent_list: AgentList,
        max_turns: int = 20,
        stopping_function: Optional[Callable] = None,
        next_statement_question: Optional[QuestionBase] = None,
        next_speaker_generator: Optional[Callable] = None,
        verbose: bool = False,
        per_round_message_template: Optional[str] = None,
        conversation_index: Optional[int] = None,
        cache=None,
        default_model: Optional["Model"] = None,
    ):
        if not isinstance(agent_list, AgentList):
            raise ConversationValueError("agent_list must be an AgentList")
        if len(agent_list) == 0:
            raise ConversationValueError("agent_list must contain at least one agent")
        if not _is_non_negative_int(max_turns):
            raise ConversationValueError("max_turns must be a non-negative integer")

        self.cache = cache
        self.per_round_message_template = per_round_message_template
        self.agent_list = agent_list
        self.verbose = verbose
        self._conversation_index = conversation_index
        self.agent_statements = AgentStatements()
        self.max_turns = max_turns

        for agent in self.agent_list:
            if not hasattr(agent, "model"):
                if default_model is not None:
                    agent.model = default_model
                else:
                    from edsl import Model
                    agent.model = Model()

        if next_statement_question is None:
            import textwrap
            base_question = textwrap.dedent(
                """\
You are {{ speaker_name }}. This is the conversation so far: {{ conversation }}
{% if round_message is not none %}
{{ round_message }}
{% endif %}
What do you say next?"""
            )
            self.next_statement_question = QuestionFreeText(
                question_text=base_question,
                question_name="dialogue",
            )
        else:
            self.next_statement_question = next_statement_question
            if next_statement_question.question_name != "dialogue":
                raise ConversationValueError(
                    "next_statement_question must have question_name='dialogue'"
                )
            if (
                per_round_message_template
                and "{{ round_message }}" not in next_statement_question.question_text
            ):
                raise ConversationValueError(
                    "If you pass in a per_round_message_template, you must include {{ round_message }} in the question_text."
                )

        if next_speaker_generator is None:
            func = default_turn_taking_generator
        else:
            func = next_speaker_generator

        self.next_speaker = speaker_closure(
            agent_list=self.agent_list, generator_function=func
        )

        if stopping_function is None:
            self.stopping_function = lambda agent_statements: False
        else:
            self.stopping_function = stopping_function

    def _should_continue(self) -> bool:
        if len(self.agent_statements) >= self.max_turns:
            return False
        return not self.stopping_function(self.agent_statements)

    def add_index(self, index) -> None:
        self._conversation_index = index

    @property
    def conversation_index(self):
        return self._conversation_index

    def _build_job(self, *, index, speaker, conversation):
        q = self.next_statement_question

        if self.per_round_message_template is None:
            round_message = None
        else:
            round_message = Template(self.per_round_message_template).render(
                {"max_turns": self.max_turns, "current_turn": index}
            )

        s = Scenario(
            {
                "speaker_name": speaker.name,
                "conversation": conversation,
                "conversation_index": self.conversation_index,
                "index": index,
                "round_message": round_message,
            }
        )
        return q.by(s).by(speaker).by(speaker.model)

    def _get_next_statement(self, *, index, speaker, conversation) -> Result:
        job = self._build_job(index=index, speaker=speaker, conversation=conversation)
        run_kwargs = {} if self.cache is None else {"cache": self.cache}
        results = job.run(**run_kwargs)
        return results[0]

    def converse(self, max_retries: int = 3, retry_delay: float = 5.0) -> None:
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 1
        ):
            raise ConversationValueError("max_retries must be a positive integer")
        if (
            not isinstance(retry_delay, (int, float))
            or isinstance(retry_delay, bool)
            or retry_delay < 0
        ):
            raise ConversationValueError("retry_delay must be a non-negative number")

        i = 0
        while self._should_continue():
            speaker = self.next_speaker()
            for attempt in range(max_retries):
                try:
                    result = self._get_next_statement(
                        index=i,
                        speaker=speaker,
                        conversation=self.agent_statements.transcript,
                    )
                    break
                except Exception as exc:
                    if not _is_transient_error(exc):
                        raise
                    if attempt == max_retries - 1:
                        if hasattr(exc, "add_note"):
                            exc.add_note(
                                f"Conversation turn {i} failed after "
                                f"{max_retries} attempts"
                            )
                        raise
                    if self.verbose:
                        print(
                            f"Transient error on attempt "
                            f"{attempt + 1}/{max_retries}: {exc}"
                        )
                    time.sleep(retry_delay)
            next_statement = AgentStatement(statement=result)
            self.agent_statements.append(next_statement)
            if self.verbose:
                print(f"'{speaker.name}': {next_statement.text}")
            i += 1

    def to_dict(self):
        return {
            "agent_list": self.agent_list.to_dict(),
            "max_turns": self.max_turns,
            "verbose": self.verbose,
            "agent_statements": [d.to_dict() for d in self.agent_statements],
            "conversation_index": self.conversation_index,
        }

    @classmethod
    def from_dict(cls, data):
        agent_list = AgentList.from_dict(data["agent_list"])
        return cls(
            agent_list=agent_list,
            max_turns=data["max_turns"],
            verbose=data["verbose"],
            conversation_index=data["conversation_index"],
        )

    def to_results(self) -> Results:
        return Results(data=[s.statement for s in self.agent_statements])

    def summarize(self) -> Scenario:
        transcript = (
            self.to_results()
            .select("agent.agent_name", "answer.dialogue")
            .to_list()
            if self.agent_statements
            else []
        )
        return Scenario(
            {
                "num_agents": len(self.agent_list),
                "max_turns": self.max_turns,
                "conversation_index": self.conversation_index,
                "transcript": transcript,
                "number_of_agent_statements": len(self.agent_statements),
            }
        )


class ConversationList:
    """Runs multiple conversations in parallel using threads.

    Each conversation blocks on its own fetch() calls, so they proceed
    independently without a shared event loop.
    """

    def __init__(self, conversations: list[Conversation]):
        if not isinstance(conversations, list):
            raise ConversationValueError("conversations must be a list")
        if not all(isinstance(c, Conversation) for c in conversations):
            raise ConversationValueError(
                "conversations must contain only Conversation objects"
            )
        self.conversations = conversations
        for i, conversation in enumerate(self.conversations):
            conversation.add_index(i)

    def run(self, max_workers: Optional[int] = None) -> None:
        """Run conversations concurrently and propagate worker failures.

        Args:
            max_workers: Maximum number of conversations to run at once. When
                omitted, ``ThreadPoolExecutor`` chooses its standard default.

        Raises:
            RuntimeError: If a conversation fails. The original exception is
                retained as the cause and the message identifies its index.
        """
        if max_workers is not None and (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers < 1
        ):
            raise ConversationValueError("max_workers must be a positive integer")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conversation = {
                executor.submit(conversation.converse): conversation
                for conversation in self.conversations
            }
            for future in as_completed(future_to_conversation):
                conversation = future_to_conversation[future]
                try:
                    future.result()
                except Exception as exc:
                    for pending in future_to_conversation:
                        pending.cancel()
                    raise RuntimeError(
                        f"Conversation {conversation.conversation_index} failed"
                    ) from exc

    def to_dict(self) -> dict:
        return {"conversations": [c.to_dict() for c in self.conversations]}

    @classmethod
    def from_dict(cls, data):
        return cls([Conversation.from_dict(d) for d in data["conversations"]])

    def to_results(self) -> Results:
        if not self.conversations:
            return Results(data=[])
        results = self.conversations[0].to_results()
        for conv in self.conversations[1:]:
            results += conv.to_results()
        return results

    def summarize(self) -> ScenarioList:
        return ScenarioList([c.summarize() for c in self.conversations])
