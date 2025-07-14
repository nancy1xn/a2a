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
from google.adk.agents import Agent
from google.genai import types 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import os
import logging
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
import time
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def start_google_server_with_executor(host, port:int)-> None:
    load_dotenv()
    session_service = InMemorySessionService()
    APP_NAME = "game_app"
    USER_ID = "user_1"
    SESSION_ID = "session_001" 

    class MoveType(str, Enum):
        Rock = "Rock"
        Paper = "Paper"
        Scissor = "Scissor"

    def get_result() -> MoveType:
        return random.choice(list(MoveType))


    class PlayerAgentExecutor(AgentExecutor):

        def __init__(self):
            self.agent = Runner(agent=Agent(
                                model="gemini-2.0-flash",
                                name="player_agent",
                                description="play rock, paper, scissor game",
                                instruction="""You are a game agent that only replies with your move.
                                                When the user sends a message, immediately call the tool `get_result`, 
                                                and reply ONLY with the result: "Rock", "Paper", or "Scissor". Nothing else.
                                                Return the result as plain text with **no punctuation**, **no quotes**, **no extra characters**, and **no leading/trailing spaces or line breaks**.

                                                Your response MUST be:
                                                - One of the exact values: Rock, Paper, Scissor
                                                - On a single line (no newline `\\n`)
                                                - With no prefix or suffix

                                                ❌ DO NOT say things like "I choose Rock", `"Paper"`, `Scissor\n`, or `Rock!`
                                                ✅ DO say: Rock"""
    ,
                                tools=[get_result]
                                    ),
                                app_name=APP_NAME,
                                session_service=session_service)

        async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            
            session = await session_service.create_session(
                        app_name=APP_NAME,
                        user_id=USER_ID,
                        session_id=SESSION_ID
            )

            content = types.Content(role='user', parts=[types.Part(text=context.message.parts[0].root.text)])

            final_response_text = "Agent did not produce a final response." # Default

            async for event in self.agent.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        # Assuming text response in the first part
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate: 
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break  
            if final_response_text:
                await event_queue.enqueue_event(new_agent_text_message(final_response_text.strip()))
            else:
                await event_queue.enqueue_event(new_agent_text_message("No response from agent."))

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

    uvicorn.run(server.build(), host=host, port=port, log_level="debug")

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
    parser.add_argument("--port", default=8003,type=int)
    args = parser.parse_args()

    start_google_server_with_executor(args.host, args.port)

    # 啟動 server 用 thread
    # server_thread = threading.Thread(target=start_server_with_executor, args=(args.host, args.port))
    # server_thread.start()

    # 等一下讓 server 啟動完成
    # time.sleep(3)
    
    # asyncio.run(interact_with_agent(args.host, args.port))




if __name__ == '__main__':
    main()