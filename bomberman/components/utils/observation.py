import math 
import typing as t

from components.types import Coordinate, Observation
from components.utils.metrics import manhattan_distance


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_bomb_to_detonate(observation: Observation, unit_id: str) -> Coordinate or None:
    entities = observation["entities"]
    bombs = list(filter(lambda entity: entity.get("unit_id") == unit_id and entity.get("type") == "b", entities))
    bomb = next(iter(bombs or []), None)
    if bomb != None:
        return [bomb.get("x"), bomb.get("y")]
    else:
        return None


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_nearest_active_bomb(observation: Observation, unit_id: str):
    unit = observation["unit_state"][unit_id]
    unit_coords = unit['coordinates']

    entities = observation["entities"]
    bombs = list(filter(lambda entity: entity.get("type") == "b", entities))

    min_distance, nearest_bomb = +math.inf, None
    for bomb in bombs:
        bomb_coords = [bomb['x'], bomb['y']]
        bomb_distance = manhattan_distance(unit_coords, bomb_coords)
        if bomb_distance < min_distance:
            min_distance = bomb_distance
            nearest_bomb = bomb

    return nearest_bomb


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_unit_activated_bombs(observation: Observation, unit_id: str):
    entities = observation["entities"]
    unit_bombs = list(filter(lambda entity: entity["type"] == "b" and entity["unit_id"] == unit_id, entities))
    return unit_bombs
    

"""
Obstacle definitions: 
    a. Wooden Block: {"created":0, "x":10, "y":1, "type":"w", "hp":1}
    b. Ore Block: {"created":0, "x":0, "y":13, "type":"o", "hp":3}
    c. Metal Block: {"created":0, "x":3, "y":7, "type":"m"}
"""
def get_obtacles(observation: t.Dict):
    entities = observation["entities"]
    obstacles = list(filter(lambda entity: entity.get("type") in ["w", "o", "m"], entities))
    return obstacles


def get_nearest_obstacle(observation: Observation, coords: Coordinate):
    entities = observation["entities"]
    obstacles = list(filter(lambda entity: entity.get("type") in ["w", "o", "m"], entities))

    min_distance, nearest_obstacle = +math.inf, None
    for obstacle in obstacles:
        obstacle_coords = [obstacle['x'], obstacle['y']]
        obstacle_distance = manhattan_distance(coords, obstacle_coords)
        if obstacle_distance < min_distance:
            min_distance = obstacle_distance
            nearest_obstacle = obstacle

    return nearest_obstacle


def get_nearest_obstacle(observation: Observation, coords: Coordinate):
    entities = observation["entities"]
    obstacles = list(filter(lambda entity: entity.get("type") in ["w", "o", "m"], entities))

    min_distance, nearest_obstacle = +math.inf, None
    for obstacle in obstacles:
        obstacle_coords = [obstacle['x'], obstacle['y']]
        obstacle_distance = manhattan_distance(coords, obstacle_coords)
        if obstacle_distance < min_distance:
            min_distance = obstacle_distance
            nearest_obstacle = obstacle

    return nearest_obstacle


def get_nearest_powerup(observation: Observation, coords: Coordinate):
    entities = observation["entities"]
    powerups = list(filter(lambda entity: entity.get("type") in ["fp", "bp"], entities))

    min_distance, nearest_powerup = +math.inf, None
    for powerup in powerups:
        powerup_coords = [powerup['x'], powerup['y']]
        powerup_distance = manhattan_distance(coords, powerup_coords)
        if powerup_distance < min_distance:
            min_distance = powerup_distance
            nearest_powerup = powerup

    return nearest_powerup


def get_nearest_opponent(observation: Observation, current_agent_id: str, coords: Coordinate):
    min_distance, nearest_opponent, nearest_opponent_coords = +math.inf, None, None
    for observed_agent_id, observed_agent_config in observation["agents"].items():
        if observed_agent_id != current_agent_id:
            for unit_id in observed_agent_config['unit_ids']:
                unit_config = observation['unit_state'][unit_id]
                if unit_config['hp'] == 0:
                    continue
                opponent_coords = [unit_config['coordinates'][0], unit_config['coordinates'][1]]
                opponent_distance = manhattan_distance(coords, opponent_coords)
                if opponent_distance < min_distance:
                    min_distance = opponent_distance
                    nearest_opponent = unit_id
                    nearest_opponent_coords = opponent_coords
    return nearest_opponent, nearest_opponent_coords

