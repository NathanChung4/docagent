"""Tool ABC and the validation-error type the agent loop reacts to.

The shape matches Anthropic's tool-use schema (name + description + input_schema)
so domain tools can be registered with the Claude API without translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class ToolValidationError(ValueError):
    """Raised by Tool.run() when inputs violate the tool's spec.

    The agent loop catches this and surfaces the message to Claude as a
    tool_result with `is_error=True`, so the model can self-correct on the
    next turn instead of silently writing a bad artifact.
    """


class Tool(ABC):
    """Base class for an agent-invocable tool.

    Subclasses set name, description, and input_schema as class attributes and
    implement run(). The shape matches Anthropic's tool-use payload directly.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    # ClassVar so subclasses get a fresh dict per class rather than sharing
    # a mutable instance attribute (the usual mutable-default footgun).
    input_schema: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with validated kwargs and return a JSON-serializable dict.

        Raise ToolValidationError for bad inputs so the agent can self-correct
        in the next turn.
        """

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Render this tool in the format Anthropic's tools= parameter expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
