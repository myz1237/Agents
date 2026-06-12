import uuid

from agent import langfuse_client, learn_run_agent_with_stream, run_agent
from utils import print_usage


def agent_runner():
    session_id = f"run-{str(uuid.uuid4())}"
    start_message = "把 test_replace.py 里的 hello 改成 hi"
    usages, _, _ = run_agent(start_message, session_id)
    print_usage(usages)
    langfuse_client.flush()


def test():
    start_message = "你好"
    learn_run_agent_with_stream(user_message=start_message, session_id=1)


if __name__ == "__main__":
    test()
