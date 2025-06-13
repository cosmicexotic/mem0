import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm
from mem0.memory.main import Memory

load_dotenv()


# Update custom instructions
custom_instructions =""" 
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

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
            }
        }
    },
}

config_graph = {
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
            }
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

class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        self.mem0_client = Memory.from_config(config)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(message, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i:i+self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item['conversation']
        speaker_a = conversation['speaker_a']
        speaker_b = conversation['speaker_b']

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)


        conv_keys = list(conversation.keys())
        # get all the session keys
        conv_keys = [key for key in conv_keys if not (key in ['speaker_a', 'speaker_b'] or "date" in key or "timestamp" in key)]

        for key in tqdm(conv_keys, total=len(conv_keys), desc=f"Processing session"):
            if key in ['speaker_a', 'speaker_b'] or "date" in key or "timestamp" in key:
                continue
            
            # session timestamp
            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]

            print(timestamp)
            if not timestamp:
                raise ValueError(f"Timestamp not found for key: {key}")

            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat['speaker'] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat['speaker'] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            # self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A")
            # self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B")

            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A")
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B")
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")
        
    def process_conversation_longmemeval(self, item, idx):
        conversation = item['haystack_sessions']
        conv_dates = item['haystack_dates']
        speaker_a = "user"
        speaker_b = "assistant"

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)


        for session, timestamp in tqdm(zip(conversation, conv_dates), total=len(conversation), desc=f"Processing conversation"):
            chats = session
            print(timestamp)
            if not timestamp:
                raise ValueError(f"Timestamp not found for key: {chats}")

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat['role'] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['content']}"})
                elif chat['role'] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['content']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['role']}")

            # add memories for the two users on different threads
            # self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A")
            # self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B")

            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for user")
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for assistant")
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()

