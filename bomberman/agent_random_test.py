import asyncio
import random
import os
import time
import typing

from components.environment.config import (
    ACTIONS, 
    FWD_MODEL_CONNECTION_RETRIES, 
    FWD_MODEL_CONNECTION_DELAY
)
from components.environment.state import GameState

GAME_CONNECTION_URI = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

class BaselineAgent():
    def __init__(self):
        self._client = GameState(GAME_CONNECTION_URI)
        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

        # any initialization code can go here
        # -----------------------------------
        state = self._client._state
        if state:
            n_actions = len(ACTIONS)
            n_observations = state.get('world', {}).get('height') * state.get('world', {}).get('width')
            print(f"World dimensions: {n_actions} x {n_observations}")
        # -----------------------------------

    def _get_bomb_to_detonate(self, unit) -> typing.Union[int, int] or None:
        entities = self._client._state.get("entities")
        bombs = list(filter(lambda entity: entity.get(
            "unit_id") == unit and entity.get("type") == "b", entities))
        bomb = next(iter(bombs or []), None)
        if bomb != None:
            return [bomb.get("x"), bomb.get("y")]
        else:
            return None

    async def _on_game_tick(self, tick_number, game_state):

        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

        for unit_id in my_units:

            action = random.choice(ACTIONS)

            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action == "detonate":
                bomb_coordinates = self._get_bomb_to_detonate(unit_id)
                if bomb_coordinates != None:
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, unit_id)
            elif action == "idle":
                # no-op
                continue
            else:
                print(f"Unhandled action: {action} for unit {unit_id}")


def main():
    print("Running random agent")
    for retry in range(FWD_MODEL_CONNECTION_RETRIES):
        try:
            BaselineAgent()
        except:
            print(f"Retrying to connect with {retry} attempt...")
            time.sleep(FWD_MODEL_CONNECTION_DELAY)
            continue
        break

main()
