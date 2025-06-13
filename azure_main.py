import os
from mem0 import Memory
import pprint
# os.environ["OPENAI_API_KEY"] = "your-api-key" # used for embedding model


os.environ["LLM_AZURE_OPENAI_API_KEY"] = "FjfmeNsmd6aBBbCOWLb4sl8RU0057djGvmGcvzhqYrOkUtifGvd0JQQJ99BEACHYHv6XJ3w3AAAAACOGIlEZ"
os.environ["LLM_AZURE_DEPLOYMENT"] = "gpt-4o-mini"
os.environ["LLM_AZURE_ENDPOINT"] = "https://123s-mann562s-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
os.environ["LLM_AZURE_API_VERSION"] = "2025-01-01-preview"

config = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": "gpt-4o-mini",
                  "api_version": "2025-01-01-preview",
                  "azure_endpoint": "https://123s-mann562s-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                  "api_key": "FjfmeNsmd6aBBbCOWLb4sl8RU0057djGvmGcvzhqYrOkUtifGvd0JQQJ99BEACHYHv6XJ3w3AAAAACOGIlEZ",
              }
        }
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": "text-embedding-3-small",
            "azure_kwargs": {
                "azure_deployment": "text-embedding-3-small",
                "api_version": "2023-05-15",
                "azure_endpoint": "https://123s-mann562s-eastus2.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15",
                "api_key": "FjfmeNsmd6aBBbCOWLb4sl8RU0057djGvmGcvzhqYrOkUtifGvd0JQQJ99BEACHYHv6XJ3w3AAAAACOGIlEZ",
            },
            "embedding_dims": 1536,
        }
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "mem0graph",
        },
    },
}

m = Memory.from_config(config)

messages = [
    {
        "role": "user",
        "content": "I'm planning to watch a movie tonight. Any recommendations?",
    },
    {
        "role": "assistant",
        "content": "How about a thriller movies? They can be quite engaging.",
    },
    {
        "role": "user",
        "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
    },
    {
        "role": "assistant",
        "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
    },
]
result = m.add(messages, user_id="alice", metadata={"category": "movies"})

query = "What movies can I watch tonight?"
pprint.pprint(result)
# pprint.pprint(m.search(query, user_id="alice"))

for result in m.search("what does alice love?", user_id="alice")["results"]:
    print(result["memory"], result["score"])