import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  

def start_judge_server_with_executor(player_a_url:str, player_b_url:str, host, port:int)-> None:

    class JudgeAgent:
        async def get_player_move(self, base_url_player)->str:
            async with httpx.AsyncClient() as httpx_client:
                # Initialize A2ACardResolver
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url_player,
                    # agent_card_path uses default, extended_agent_card_path also uses default
                )
                # --8<-- [end:A2ACardResolver]

                # Fetch Public Agent Card and Initialize Client
                final_agent_card_to_use: AgentCard | None = None

                _public_card = (
                    await resolver .get_agent_card()
                )  
                final_agent_card_to_use  = _public_card

                # --8<-- [start:send_message]
                client = A2AClient(
                    httpx_client=httpx_client, agent_card=final_agent_card_to_use
                )


                send_message_payload: dict[str, Any] = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': 'What move will you do?'}
                        ],
                        'messageId': uuid4().hex,
                    },
                }
                request = SendMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
                )

                response = await client.send_message(request)
                r=response.model_dump(mode='json', exclude_none=True)

                logger.info("==== context start ====")
                logger.info("Message text: %s",response)
                logger.info("JSON Message text: %s",r)
                logger.info("==== context end ====")
                
                move= r['result']['parts'][0]['text']

                return move

        async def invoke(self, move_a, move_b):    
                if move_a == "Scissor" and move_b =="Scissor":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nIt's a tie, please play the next round"
                
                if move_a == "Scissor" and move_b =="Rock":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nB is the winner"
                
                if move_a == "Scissor" and move_b =="Paper":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nA is the winner"
                
                if move_a == "Rock" and move_b =="Rock":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nIt's a tie, please play the next round"
                
                if move_a == "Rock" and move_b =="Scissor":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nA is the winner"
                
                if move_a == "Rock" and move_b =="Paper":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nB is the winner"
                
                if move_a == "Paper" and move_b =="Paper":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nIt's a tie, please play the next round"
                
                if move_a == "Paper" and move_b =="Scissor":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nB is the winner"
                
                if move_a == "Paper" and move_b =="Rock":
                    return f"Player A: {move_a}\nPlayer B: {move_b} \nA is the winner"
                


    class JudgeAgentExecutor(AgentExecutor):
        """Test AgentProxy Implementation."""
        
        def __init__(self):
            self.agent = JudgeAgent()

        async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            move_a = await self.agent.get_player_move(player_a_url)
            move_b = await self.agent.get_player_move(player_b_url)
            result = await self.agent.invoke(move_a, move_b)
            await event_queue.enqueue_event(new_agent_text_message(result))

        async def cancel(
            self, context: RequestContext, event_queue: EventQueue
        ) -> None:
            raise Exception('cancel not supported')
    
    skill_judge = AgentSkill(
        id='rock_paper_scissor_judge',
        name='Returns the winner of rock paper scissor',
        description='just returns winner name',
        tags=['game', 'rps'],
        examples=['Player A'],
    )


    public_agent_card = AgentCard(
        name='judge of rock paper scissor',
        description='Just a rock paper scissor judge',
        url=f"http://{host}:{port}/", 
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_judge],  
        supportsAuthenticatedExtendedCard=True,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=JudgeAgentExecutor(),
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
    parser.add_argument("--peer_a_url", default="http://localhost:8001/")
    parser.add_argument("--peer_b_url", default="http://localhost:8002/")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8005",type=int)
    args = parser.parse_args()
    
    # 啟動 server 用 thread
    server_thread = threading.Thread(target=start_judge_server_with_executor, args=(args.peer_a_url, args.peer_b_url, args.host, args.port))
    server_thread.start()

    # 等一下讓 server 啟動完成
    import time
    time.sleep(1)
    
    asyncio.run(interact_with_agent(args.host, args.port))


if __name__ == '__main__':
    main()
