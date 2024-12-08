import json

class ActionContext:
    def __init__(self, player_id, player_name, building, websocket, action_id, action_name):
        self.player_id = player_id
        self.player_name = player_name
        self.building = building
        self.websocket = websocket
        self.action_id = action_id
        self.action_name = action_name

    async def send_result(self, payload, payment=0):
        """
        Sends the result of the action back to the server.

        :param payload: The payload to send back as the result.
        """
        action_result = {
            "type": "actionResult",
            "data": {
                "playerID": self.player_id,
                "action": self.action_name,
                "actionID": self.action_id,
                "payload": payload,
                "payment": payment,
            }
        }
        await self.websocket.send(json.dumps(action_result))
