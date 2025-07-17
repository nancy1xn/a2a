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
import json

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
import asyncio
import threading
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

async def start_server_with_executor(host, port:int)-> None:

    class MoveType(str, Enum):
        Rock = "Rock"
        Paper = "Paper"
        Scissor = "Scissor"
    
    class PlayerAgent:
        """Rock Paper Scissor Agent."""
        async def invoke(self) -> str:
            client = OpenAI()

            tools = [{
                "type": "function",
                "name": "make_move",
                "description": "Play Rock, Paper, Scissor by randomly choosing and returning one of the move from Movetype.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": ["string", "null"],
                            "enum": ["Rock", "Paper", "Scissor"],
                            "description": "Movetype will be returned in."
                        }
                    },
                        "required": [
                        "move"
                        ],
                    "additionalProperties": False
                },
                "strict": True
            }]

            input_messages = [{"role": "user", "content": "Rock Paper Scissor!"}]

            response = client.responses.create(
                model="gpt-4.1",
                input=input_messages,
                tools=tools,
            )
            json_response =json.loads(response.output[0].arguments)
            return json_response["move"]

    # version1
    #class MoveType(str, Enum):
    #    Rock = "Rock"
    #    Paper = "Paper"
    #    Scissor = "Scissor"
    # class PlayerAgent:
    #     """Rock Paper Scissor Agent."""

    #     async def invoke(self) -> MoveType:
    #         return random.choice(list(MoveType))

    # version2 
    # class Move(BaseModel):
    #     move: MoveType
  
    # class PlayerAgent:
    #     """Rock Paper Scissor Agent."""
    #     async def invoke(self) -> MoveType:
    #         model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    #         structured_model = model.with_structured_output(Move)
    #         system = """You are a player that play Rock, Paper, Scissor. Please randomly choose and return one of the move from Movetype without any other word.
    #                     DO NOT CHOOSE THE SAME MOVE EVERYTIME. For example: (1)"Rock" (2)"Paper" """
    #         llm_response = structured_model.invoke(
    #         [
    #             SystemMessage(content=system),
    #             HumanMessage(content="Rock, Paper, Scissor, please show your move randomly and change each time."),
    #             ]
    #         )
    #         return llm_response.move


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
    config = uvicorn.Config(server.build(), host=host, port=port, log_level="info", lifespan="off")
    server = uvicorn.Server(config)
    await server.serve()
    # uvicorn.run(server.build(), host=host, port=port)


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



async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8001", type=int)
    args = parser.parse_args()

    asyncio.create_task(start_server_with_executor(args.host, args.port))

    # # 啟動 server 用 thread
    # server_thread = threading.Thread(target=start_server_with_executor, args=(args.host, args.port))
    # server_thread.start()

    # 等一下讓 server 啟動完成
    import time
    time.sleep(1)


    await asyncio.sleep(5)
    #等client完成才算完成 所以要await他
    await interact_with_agent(args.host, args.port)



if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--host", default="localhost")
    # parser.add_argument("--port", default="8001", type=int)
    # args = parser.parse_args()

    # loop = asyncio.get_running_loop()
    # loop.create_task(start_server_with_executor(args.host, args.port)) 

    asyncio.run(main())


    
