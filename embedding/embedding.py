from pydantic import BaseModel
from typing import List, Dict, Any

class EmbeddingModel(BaseModel):
    name: str
    dimensions: int
    provider: str
    metadata: Dict[str, Any] = {}

class EmbeddingProvider(BaseModel):
    name: str
    models: List[EmbeddingModel]

    def get_model(self, name: str) -> EmbeddingModel:
        for model in self.models:
            if model.name == name:
                return model
        raise ValueError(f"Model {name} not found for provider {self.name}")

# OpenAI Embedding Models
openai_provider = EmbeddingProvider(
    name="openai",
    models=[
        EmbeddingModel(
            name="text-embedding-3-small",
            dimensions=1536,
            provider="openai",
            metadata={"type": "small"}
        ),
        EmbeddingModel(
            name="text-embedding-3-large",
            dimensions=3072,
            provider="openai",
            metadata={"type": "large"}
        ),
        EmbeddingModel(
            name="text-embedding-ada-002",
            dimensions=1536,
            provider="openai",
            metadata={"type": "ada"}
        ),
    ]
)

# You can add other providers here, for example:
# cohere_provider = ...
# voyage_provider = ...

ALL_PROVIDERS = [openai_provider]

def get_provider(name: str) -> EmbeddingProvider:
    for provider in ALL_PROVIDERS:
        if provider.name == name:
            return provider
    raise ValueError(f"Provider {name} not found")

def get_embedding_model(provider_name: str, model_name: str) -> EmbeddingModel:
    provider = get_provider(provider_name)
    return provider.get_model(model_name)
