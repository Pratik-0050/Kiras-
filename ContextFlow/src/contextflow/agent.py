# contextflow/agent.py
"""
Step 19: Simple ContextFlow agent (no planning).
Step 20: Basic tool calling (safe tools only, no shell execution).
Step 21: Intelligent tool-result compaction.

Agent runs one deterministic turn per run() call: retrieve relevant
context for the user request, assemble the prompt within the token
budget, call the configured LLMClient, append both turns to the
conversation history, and trigger the compaction pipeline when token
pressure requires it. Repeated run() calls accumulate history in the
same Conversation.

When the model replies with ```tool fenced-JSON blocks for registered
tools, the agent executes them, reduces large outputs via
ToolResultProcessor before adding the observations (full originals stay
in the archive and message metadata for later retrieval), and continues
the loop (bounded by max_tool_rounds) before recording history. Tool
observations are ordinary Messages: they count toward the token budget
and can trigger compaction like any other turn.

This is a straightforward request/response loop -- not an autonomous
agent: no goals, no planning. Failures propagate (LLMError from the
model, StoreError from persistence) instead of being retried or hidden;
a failed turn leaves history untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from .conversation import Conversation
from .hybrid import AssembledContext, ContextAssembler
from .llm import LLMClient, LLMResponse
from .message import Message
from .pipeline import ContextPipeline, PipelineResult
from .retriever import KeywordRetriever, RetrievalResult, Retriever
from .tools import (
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolResultProcessor,
    parse_tool_calls,
)


@dataclass
class AgentResult:
    """Outcome of one Agent.run() turn.

    Attributes:
        response:            Final model reply text.
        llm_response:        Last LLMResponse (usage, latency, model name).
        assistant_message:   Final reply stored in the conversation history.
        prompt_tokens:       Assembled context tokens sent, summed over all
                             model calls in the turn.
        completion_tokens:   Reply tokens reported by the model, summed over
                             all calls in the turn.
        total_tokens:        prompt_tokens + completion_tokens.
        compaction_occurred: True when the turn ended with real compaction
                             (triggered, committed, messages summarized).
        pipeline_result:     PipelineResult from the post-turn check
                             (triggered or clean skip).
        assembled:           Final assembled prompt context sent to the model.
        retrieved:           Retrieval hits used during the turn (all rounds).
        tool_calls:          Tool executions in order (empty without tools).
        tool_rounds:         Loop iterations that executed tools.
        tools_truncated:     True when tool requests remained after
                             max_tool_rounds was exhausted.
        turns:               Conversation message count after the turn.
    """
    response: str
    llm_response: LLMResponse
    assistant_message: Message
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    compaction_occurred: bool = False
    pipeline_result: Optional[PipelineResult] = None
    assembled: Optional[AssembledContext] = None
    retrieved: List[RetrievalResult] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_rounds: int = 0
    tools_truncated: bool = False
    turns: int = 0

    def __str__(self) -> str:
        sep = "-" * 52
        compacted = "yes" if self.compaction_occurred else "no"
        preview = self.response[:80] + ("..." if len(self.response) > 80 else "")
        return "\n".join([
            sep,
            "Agent Result",
            sep,
            f"  Response   : {preview}",
            f"  Tokens     : {self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion (= {self.total_tokens})",
            f"  Compacted  : {compacted}",
            f"  Retrieved  : {len(self.retrieved)}",
            f"  Tools      : {len(self.tool_calls)} call(s) "
            f"in {self.tool_rounds} round(s)",
            f"  Turns      : {self.turns} message(s) in history",
            sep,
        ])


class Agent:
    """Simple agent that uses ContextFlow automatically.

    One run() call performs one turn:
      1. Retrieve  -- relevant history for the request (keyword by default).
      2. Assemble  -- system + protected + retrieved + recent + request
                      packed into the token budget.
      3. Complete  -- LLMClient answers using the assembled context.
      4. Tools     -- while the reply requests registered tools (bounded
                      by max_tool_rounds): execute, reduce large outputs
                      via the result processor, append observations,
                      reassemble, and ask again.
      5. Record    -- user request, assistant replies, and tool
                      observations appended to history (only after success;
                      failed turns change nothing).
      6. Compact   -- ContextPipeline runs; compaction happens only when
                      its trigger threshold is exceeded.

    Args:
        llm:            LLMClient answering turns (required).
        conversation:   History holder. Defaults to a new Conversation.
                        Its max_tokens is the assembly budget fallback.
        retriever:      Retriever for stage 1. Defaults to KeywordRetriever().
        assembler:      ContextAssembler for stage 2. Built per run from
                        max_tokens / keep_recent when not given.
        pipeline:       ContextPipeline for stage 6. Defaults to
                        ContextPipeline().
        system:         System instructions sent with every turn.
        max_tokens:     Assembly budget. Defaults to None, meaning the
                        conversation's max_tokens at run time.
        keep_recent:    Recent messages kept by the per-run assembler
                        (non-negative int, default 5). Ignored when a
                        custom assembler is given.
        retrieval_top_k: Hits fetched in stage 1 (positive int, default 5).
        tools:          ToolRegistry or list of Tools available in stage 4.
                        Defaults to an empty registry (tool blocks in
                        replies are treated as plain text).
        max_tool_rounds: Loop iterations executing tools per turn
                         (non-negative int, default 3). 0 disables tool
                         execution.
        result_processor: ToolResultProcessor reducing large tool outputs
                         before they enter active context. Defaults to a
                         default-configured processor; pass False to
                         disable processing (raw outputs recorded).

    Usage::

        agent = Agent(llm=OpenAILLMClient(), system="You are concise.",
                      tools=[ReadFileTool(root)])
        result = agent.run("summarize notes.txt")
        print(result.response, result.total_tokens, result.compaction_occurred)
    """

    def __init__(
        self,
        llm: LLMClient,
        conversation: Optional[Conversation] = None,
        retriever: Optional[Retriever] = None,
        assembler: Optional[ContextAssembler] = None,
        pipeline: Optional[ContextPipeline] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        keep_recent: int = 5,
        retrieval_top_k: int = 5,
        tools: Union[ToolRegistry, Sequence[Tool], None] = None,
        max_tool_rounds: int = 3,
        result_processor: Union[ToolResultProcessor, bool, None] = None,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError(
                f"llm must be an LLMClient instance, got {type(llm).__name__!r}."
            )
        if conversation is not None and not isinstance(conversation, Conversation):
            raise TypeError(
                "conversation must be a Conversation instance or None, "
                f"got {type(conversation).__name__!r}."
            )
        if retriever is not None and not isinstance(retriever, Retriever):
            raise TypeError(
                "retriever must be a Retriever instance or None, "
                f"got {type(retriever).__name__!r}."
            )
        if assembler is not None and not isinstance(assembler, ContextAssembler):
            raise TypeError(
                "assembler must be a ContextAssembler instance or None, "
                f"got {type(assembler).__name__!r}."
            )
        if pipeline is not None and not isinstance(pipeline, ContextPipeline):
            raise TypeError(
                "pipeline must be a ContextPipeline instance or None, "
                f"got {type(pipeline).__name__!r}."
            )
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a string or None, got {type(system).__name__!r}."
            )
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                raise TypeError(
                    "max_tokens must be a positive integer or None, "
                    f"got {type(max_tokens).__name__!r}."
                )
            if max_tokens <= 0:
                raise ValueError(
                    f"max_tokens must be a positive integer or None, got {max_tokens}."
                )
        if not isinstance(keep_recent, int) or isinstance(keep_recent, bool):
            raise TypeError(
                "keep_recent must be a non-negative integer, "
                f"got {type(keep_recent).__name__!r}."
            )
        if keep_recent < 0:
            raise ValueError(
                f"keep_recent must be a non-negative integer, got {keep_recent}."
            )
        if not isinstance(retrieval_top_k, int) or isinstance(retrieval_top_k, bool):
            raise TypeError(
                "retrieval_top_k must be a positive integer, "
                f"got {type(retrieval_top_k).__name__!r}."
            )
        if retrieval_top_k <= 0:
            raise ValueError(
                f"retrieval_top_k must be a positive integer, got {retrieval_top_k}."
            )
        if tools is None:
            registry = ToolRegistry()
        elif isinstance(tools, ToolRegistry):
            registry = tools
        elif isinstance(tools, (list, tuple)):
            registry = ToolRegistry(list(tools))
        else:
            raise TypeError(
                "tools must be a ToolRegistry, a list of Tools, or None, "
                f"got {type(tools).__name__!r}."
            )
        if not isinstance(max_tool_rounds, int) or isinstance(max_tool_rounds, bool):
            raise TypeError(
                "max_tool_rounds must be a non-negative integer, "
                f"got {type(max_tool_rounds).__name__!r}."
            )
        if max_tool_rounds < 0:
            raise ValueError(
                f"max_tool_rounds must be a non-negative integer, got {max_tool_rounds}."
            )
        if result_processor is None:
            processor: Optional[ToolResultProcessor] = ToolResultProcessor()
        elif result_processor is False:
            processor = None
        elif isinstance(result_processor, ToolResultProcessor):
            processor = result_processor
        else:
            raise TypeError(
                "result_processor must be a ToolResultProcessor, False, or None, "
                f"got {type(result_processor).__name__!r}."
            )

        self.llm: LLMClient = llm
        self.conversation: Conversation = (
            conversation if conversation is not None else Conversation(name="Agent Session")
        )
        self.retriever: Retriever = retriever if retriever is not None else KeywordRetriever()
        self.assembler: Optional[ContextAssembler] = assembler
        self.pipeline: ContextPipeline = pipeline if pipeline is not None else ContextPipeline()
        self.system: Optional[str] = system
        self.max_tokens: Optional[int] = max_tokens
        self.keep_recent: int = keep_recent
        self.retrieval_top_k: int = retrieval_top_k
        self.tools: ToolRegistry = registry
        self.max_tool_rounds: int = max_tool_rounds
        self.result_processor: Optional[ToolResultProcessor] = processor

    # ------------------------------------------------------------------
    # History + single-turn loop
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Message]:
        """Current conversation history (safe copy)."""
        return self.conversation.get_messages()

    def register_tool(self, tool: Tool) -> None:
        """Register a tool on this agent's registry (see ToolRegistry)."""
        self.tools.register(tool)

    def _view(self, extra: List[Message]) -> Conversation:
        """Read-only view of history plus pending turn messages."""
        view = Conversation(
            name=self.conversation.name,
            max_tokens=self.conversation.max_tokens,
            warn_at=self.conversation.warn_at,
            compact_at=self.conversation.compact_at,
        )
        for message in self.conversation.get_messages() + extra:
            view.add(message)
        return view

    def _observation_message(self, result: ToolResult) -> Message:
        """Build the tool observation message, reducing large outputs.

        The reduced text becomes the message content (what the model sees
        and what token accounting charges); the full original travels in
        metadata (and the processor archive) for later retrieval.
        """
        if self.result_processor is None:
            return Message(
                role="tool",
                content=result.to_message_content(),
                metadata={"type": "tool_result", "tool": result.tool_name,
                          "ok": result.ok},
            )
        processed = self.result_processor.process(result)
        metadata = {
            "type": "tool_result",
            "tool": result.tool_name,
            "ok": result.ok,
            "call_id": processed.call_id,
            "reduced": processed.was_reduced,
            "original_tokens": processed.original_tokens,
        }
        if processed.was_reduced:
            metadata["full_output"] = processed.original
        return Message(
            role="tool",
            content=processed.to_message_content(),
            metadata=metadata,
        )

    def run(self, request: Union[str, Message]) -> AgentResult:
        """Run one turn: retrieve, assemble, complete, tools, record, compact.

        Tool blocks (```tool fenced JSON) in a reply are executed against
        the registry and fed back until the model answers plainly or
        max_tool_rounds is exhausted. Everything is buffered and committed
        to history only after success.

        Args:
            request: Current user request (non-blank string or Message).

        Returns:
            AgentResult with the final reply, summed token statistics,
            tool calls, and whether compaction occurred.

        Raises:
            TypeError:  If *request* is not a string or Message.
            ValueError: If *request* is blank, or no token budget is
                        available (neither max_tokens nor
                        conversation.max_tokens is set).
            LLMError:   If a model call fails (history untouched).
        """
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("request must not be blank.")
            request_msg = Message(role="user", content=request)
        elif isinstance(request, Message):
            if request.role != "user":
                raise ValueError(
                    f"request messages must use role 'user', got {request.role!r}."
                )
            request_msg = request
        else:
            raise TypeError(
                f"request must be a string or Message, got {type(request).__name__!r}."
            )

        budget = self.max_tokens or self.conversation.max_tokens
        if budget is None:
            raise ValueError(
                "No token budget: pass max_tokens=... or set "
                "conversation.max_tokens=...."
            )
        assembler = self.assembler or ContextAssembler(
            max_tokens=budget, keep_recent=self.keep_recent
        )

        # Buffered turn: committed to history only after success.
        pending: List[Message] = [request_msg]
        tool_calls: List[ToolCall] = []
        retrieved_all: List[RetrievalResult] = []
        responses: List[LLMResponse] = []
        assembled_list: List[AssembledContext] = []
        tool_rounds = 0
        tools_truncated = False

        # 1-2. Retrieve relevant history, then assemble the prompt.
        retrieved = self.retriever.retrieve(
            request_msg.content, self.conversation, top_k=self.retrieval_top_k
        )
        retrieved_all.extend(retrieved)

        while True:
            assembled = assembler.assemble(request_msg, self._view(pending), retrieved)
            assembled_list.append(assembled)

            # 3. Ask the model (assembled context already ends with the request).
            llm_response = self.llm.complete(context=assembled, system=self.system)
            responses.append(llm_response)
            pending.append(Message(role="assistant", content=llm_response.content))

            # 4. Tool loop: execute requested blocks, then ask again.
            calls = parse_tool_calls(llm_response.content)
            if not calls or len(self.tools) == 0 or self.max_tool_rounds == 0:
                break
            if tool_rounds >= self.max_tool_rounds:
                tools_truncated = True
                break
            for parsed in calls:
                if parsed.error is not None:
                    result = ToolResult(
                        tool_name=parsed.name, ok=False, error=parsed.error)
                else:
                    result = self.tools.execute(parsed.name, parsed.arguments)
                tool_calls.append(ToolCall(
                    name=parsed.name or result.tool_name,
                    arguments=parsed.arguments,
                    result=result,
                ))
                pending.append(self._observation_message(result))
            tool_rounds += 1
            retrieved = self.retriever.retrieve(
                request_msg.content, self._view(pending), top_k=self.retrieval_top_k
            )
            retrieved_all.extend(retrieved)

        # 5. Record the turn only after success.
        for message in pending:
            self.conversation.add(message)
        assistant_msg = pending[-1]

        # 6. Compact when pressure requires it (pipeline gates internally).
        pipeline_result = self.pipeline.run(self.conversation)
        compaction = pipeline_result.compaction
        compaction_occurred = bool(
            pipeline_result.triggered
            and pipeline_result.committed
            and compaction is not None
            and compaction.was_needed
        )

        prompt_tokens = sum(a.total_tokens for a in assembled_list)
        completion_tokens = sum(r.output_tokens for r in responses)
        final = responses[-1]
        return AgentResult(
            response=final.content,
            llm_response=final,
            assistant_message=assistant_msg,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            compaction_occurred=compaction_occurred,
            pipeline_result=pipeline_result,
            assembled=assembled_list[-1],
            retrieved=retrieved_all,
            tool_calls=tool_calls,
            tool_rounds=tool_rounds,
            tools_truncated=tools_truncated,
            turns=self.conversation.message_count(),
        )
