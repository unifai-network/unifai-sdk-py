from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .toolkit import Toolkit

from typing import Any
from pydantic import BaseModel
from .messages import ToolkitToServerMessage, ToolkitToServerMessageType, ActionResultMessageData

class ActionResult(BaseModel):
    """
    Represents the result of an action.

    :param payload: The payload to send back as the result.
    :param payment: The actual payment amount associated with the action. Positive values indicate charges to agents,
                    while negative values indicate payments to agents. The value must not exceed the payment value
                    passed to the action handler.
    """
    payload: Any
    payment: float = 0

class ActionContext:
    """
    Represents the context of an action.
    """
    toolkit: "Toolkit"
    agent_id: int
    action_id: int
    action_name: str

    def __init__(self, toolkit: "Toolkit", agent_id: int, action_id: int, action_name: str):
        self.toolkit = toolkit
        self.agent_id = agent_id
        self.action_id = action_id
        self.action_name = action_name

    def Result(self, payload: Any, payment: float = 0) -> ActionResult:
        """
        Creates an ActionResult instance.

        :param payload: The payload to send back as the result.
        :param payment: The actual payment amount associated with the action. Positive values indicate charges to agents,
                       while negative values indicate payments to agents. The value must not exceed the payment value
                       passed to the action handler.
        :return: An instance of ActionResult.
        """
        return ActionResult(payload=payload, payment=payment)

    async def send_result(self, result: ActionResult):
        """
        Sends the result of the action back to the server.
        """
        action_result_message = ToolkitToServerMessage(
            type=ToolkitToServerMessageType.ACTION_RESULT,
            data=ActionResultMessageData(
                action=self.action_name,
                actionID=self.action_id,
                agentID=self.agent_id,
                payload=result.payload,
                payment=result.payment,
            )
        )
        await self.toolkit._ws.send(action_result_message.model_dump_json())
