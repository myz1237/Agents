import uuid

from agent import langfuse_client, run_agent
from utils import print_usage

if __name__ == "__main__":
    session_id = f"run-{str(uuid.uuid4())}"
    start_message = "把 test_replace.py 里的 hello 改成 hi"
    usages, _ = run_agent(start_message, session_id)
    print_usage(usages)
    langfuse_client.flush()
