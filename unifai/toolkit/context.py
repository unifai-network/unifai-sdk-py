from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .toolkit import Toolkit

from websockets.asyncio.client import ClientConnection
from .messages import *

class ActionContext:
    toolkit: "Toolkit"
    websocket: ClientConnection
    agent_id: int
    action_id: int
    action_name: str

    def __init__(self, toolkit: "Toolkit", websocket: ClientConnection, agent_id: int, action_id: int, action_name: str):
        self.toolkit = toolkit
        self.websocket = websocket
        self.agent_id = agent_id
        self.action_id = action_id
        self.action_name = action_name

    async def send_result(self, payload: Any, payment: float = 0):
        """
        Sends the result of the action back to the server.

        :param payload: The payload to send back as the result.
        :param payment: The actual payment amount associated with the action. Positive values are charging agents,
                       negative values are paying agents. The value must be no greater than payment value passed
                       to the action handler.
        """
        action_result_message = ToolkitToServerMessage(
            type=ToolkitToServerMessageType.ACTION_RESULT,
            data=ActionResultMessageData(
                action=self.action_name,
                actionID=self.action_id,
                agentID=self.agent_id,
                payload=payload,
                payment=payment,
            )
        )
        await self.websocket.send(action_result_message.model_dump_json())
