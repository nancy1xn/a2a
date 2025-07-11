from player import start_server_with_executor
from judge import start_judge_server_with_executor, interact_with_agent
from google_agents.google_player import start_google_server_with_executor
import threading
import argparse
import asyncio
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer_a_url", default="http://localhost:8001/")
    parser.add_argument("--peer_b_url", default="http://localhost:8002/")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8005",type=int)
    args = parser.parse_args()

    server_thread = threading.Thread(target=start_server_with_executor, args=("localhost", 8001)).start()
    server_thread = threading.Thread(target=start_server_with_executor, args=("localhost", 8002)).start()
    server_thread = threading.Thread(target=start_google_server_with_executor, args=("localhost", 8003)).start()

    server_thread = threading.Thread(target=start_judge_server_with_executor, args=(args.peer_a_url, args.peer_b_url, args.host, args.port))
    server_thread.start()
    time.sleep(1)
    asyncio.run(interact_with_agent(args.host, args.port))

if __name__ == '__main__':
    main()