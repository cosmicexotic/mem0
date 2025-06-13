# to print json, import pprint
import pprint
from mem0 import MemoryClient
client = MemoryClient(api_key="m0-1ueJFP3X8bz8rERwnGA6tjIcNGFsMUq93juuOLna")


messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]
client.add(messages, user_id="alex")
query = "What can I cook for dinner tonight?"
pprint.pprint(client.search(query, user_id="alex"))