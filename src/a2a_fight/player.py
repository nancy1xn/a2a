import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from enum import Enum
import random
import argparse
import logging

from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
import asyncio
import threading

def start_server_with_executor(host, port:int)-> None:

    class MoveType(str, Enum):
        Rock = "Rock"
        Paper = "Paper"
        Scissor = "Scissor"

    class PlayerAgent:
        """Rock Paper Scissor Agent."""

        async def invoke(self) -> MoveType:
            return random.choice(list(MoveType))

    class PlayerAgentExecutor(AgentExecutor):
        """Test AgentProxy Implementation."""

        def __init__(self):
            self.agent = PlayerAgent()

        async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            result = await self.agent.invoke()
            await event_queue.enqueue_event(new_agent_text_message(result))

        async def cancel(
            self, context: RequestContext, event_queue: EventQueue
        ) -> None:
            raise Exception('cancel not supported')

    skill_player = AgentSkill(
        id='rock_paper_scissor_player',
        name='Returns rock paper scissor',
        description='just returns rock, paper, or scissor',
        tags=['game', 'rps'],
        examples=['rock'],
    )

    public_agent_card = AgentCard(
        name='player of rock paper scissor',
        description='Just a rock paper scissor agent plays game',
        url=f"http://{host}:{port}/", 
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_player],  
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PlayerAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=host, port=port)


async def interact_with_agent(host, port:int)-> None:

    PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'

    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Get a logger instance

    # --8<-- [start:A2ACardResolver]

    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=f"http://{host}:{port}/",
            # agent_card_path uses default, 
        )
        # --8<-- [end:A2ACardResolver]

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {f"http://{host}:{port}/"}{PUBLIC_AGENT_CARD_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        # --8<-- [start:send_message]
        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        logger.info('A2AClient initialized.')

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'Rock, Paper, Scissor'}
                ],
                'messageId': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8001", type=int)
    args = parser.parse_args()

    # start_server_with_executor(args.host, args.port)

    # 啟動 server 用 thread
    server_thread = threading.Thread(target=start_server_with_executor, args=(args.host, args.port))
    server_thread.start()

    # 等一下讓 server 啟動完成
    import time
    time.sleep(1)
    
    asyncio.run(interact_with_agent(args.host, args.port))



if __name__ == '__main__':
    main()


    
