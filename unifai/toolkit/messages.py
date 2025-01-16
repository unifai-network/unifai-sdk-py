from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ServerToToolkitMessageType(Enum):
    ACTION = "action"
    TOOLKIT = "toolkit"

class ServerToToolkitMessage(BaseModel):
    type: ServerToToolkitMessageType
    data: Optional[Dict[str, Any]]

class ActionMessageData(BaseModel):
    action: str
    actionID: int
    agentID: int
    payload: Optional[Dict[str, Any] | str]
    payment: Optional[float]

class ToolkitToServerMessageType(Enum):
    REGISTER_ACTIONS = "registerActions"
    ACTION_RESULT = "actionResult"

class ToolkitToServerMessage(BaseModel):
    type: ToolkitToServerMessageType
    data: Any

class ActionDescription(BaseModel):
    description: str | dict
    payload: str | dict
    payment: str | dict

class RegisterActionsMessageData(BaseModel):
    actions: Dict[str, ActionDescription]

class ActionResultMessageData(BaseModel):
    action: str
    actionID: int
    agentID: int
    payload: Any
    payment: Optional[float]
