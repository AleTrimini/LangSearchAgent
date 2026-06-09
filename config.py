from dataclasses import dataclass


@dataclass(frozen=True)
class AgentModels:
    supervisor: str = "gpt-4o"
    researcher: str = "gpt-4o"
    summarizer: str = "gpt-4o"
    writer: str = "gpt-4o"


DEFAULT_MODELS = AgentModels()
