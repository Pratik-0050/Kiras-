# contextflow/__init__.py
"""
ContextFlow: An intelligent context compaction and conversation management engine for LLMs.
"""

__version__ = "1.0.0"

from .message      import Message, ImportanceLevel, VALID_IMPORTANCE
from .conversation import Conversation
from .status       import ContextStatus
from .compactor    import Compactor, CompactionResult
from .store        import ContextStore, StoreError
from .embeddings   import (
    EmbeddingProvider,
    BaseEmbeddingProvider,
    EmbeddingError,
    OpenAIEmbeddingProvider,
)
from .retriever    import (
    Retriever,
    BaseRetriever,
    KeywordRetriever,
    SemanticRetriever,
    VectorStoreRetriever,
    RetrievalResult,
    RetrieverError,
)
from .vectorstores import (
    VectorStore,
    BaseVectorStore,
    VectorHit,
    VectorStoreError,
    ChromaVectorStore,
)
from .hybrid      import (
    HybridRetriever,
    ContextAssembler,
    AssembledContext,
    ExcludedItem,
)
from .pipeline     import (
    ContextPipeline,
    PipelineResult,
)
from .manager      import ContextManager
from .agent        import (
    Agent,
    AgentResult,
)
from .tools import (
    Tool,
    BaseTool,
    ToolCall,
    ToolCallRequest,
    ToolError,
    ToolRegistry,
    ToolResult,
    parse_tool_calls,
    ListDirectoryTool,
    ReadFileTool,
    default_filesystem_tools,
    ProcessedToolResult,
    ToolResultProcessor,
)
from .llm          import (
    LLMClient,
    BaseLLMClient,
    LLMResponse,
    LLMError,
    OpenAILLMClient,
)
from .summarizers  import (
    Summarizer,
    BaseSummarizer,
    SummarizerError,
    PlaceholderSummarizer,
    OpenAISummarizer,
)
from .validators import (
    Validator,
    BaseValidator,
    ValidationResult,
    ValidatorError,
    HeuristicValidator,
    OpenAIValidator,
)
from .scorers import (
    PriorityScorer,
    BasePriorityScorer,
    PriorityScore,
    ScorerError,
    HeuristicPriorityScorer,
)

__all__ = [
    # Core
    "Message",
    "ImportanceLevel",
    "VALID_IMPORTANCE",
    "Conversation",
    "ContextStatus",
    "Compactor",
    "CompactionResult",
    "ContextStore",
    "StoreError",
    # Retrieval
    "Retriever",
    "BaseRetriever",
    "KeywordRetriever",
    "SemanticRetriever",
    "VectorStoreRetriever",
    "RetrievalResult",
    "RetrieverError",
    # Vector stores
    "VectorStore",
    "BaseVectorStore",
    "VectorHit",
    "VectorStoreError",
    "ChromaVectorStore",
    # Hybrid retrieval + assembly
    "HybridRetriever",
    "ContextAssembler",
    "AssembledContext",
    "ExcludedItem",
    # Pipeline
    "ContextPipeline",
    "PipelineResult",
    # Facade (simple public entry point)
    "ContextManager",
    "__version__",
    # Agent
    "Agent",
    "AgentResult",
    # Tools
    "Tool",
    "BaseTool",
    "ToolCall",
    "ToolCallRequest",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "parse_tool_calls",
    "ListDirectoryTool",
    "ReadFileTool",
    "default_filesystem_tools",
    "ProcessedToolResult",
    "ToolResultProcessor",
    # LLM adapter
    "LLMClient",
    "BaseLLMClient",
    "LLMResponse",
    "LLMError",
    "OpenAILLMClient",
    # Embeddings
    "EmbeddingProvider",
    "BaseEmbeddingProvider",
    "EmbeddingError",
    "OpenAIEmbeddingProvider",
    # Summarizers
    "Summarizer",
    "BaseSummarizer",
    "SummarizerError",
    "PlaceholderSummarizer",
    "OpenAISummarizer",
    # Validators
    "Validator",
    "BaseValidator",
    "ValidationResult",
    "ValidatorError",
    "HeuristicValidator",
    "OpenAIValidator",
    # Scorers
    "PriorityScorer",
    "BasePriorityScorer",
    "PriorityScore",
    "ScorerError",
    "HeuristicPriorityScorer",
]
