import torch
import typing as t
import numpy as np

from components.types import Map, Observation, Unit
from components.environment.config import ACTIONS, MAX_UNIT_VALUE


UNIT_DIMS = 5


"""
State and action dimensions: HYBRID approach
"""

def map_dimensions(observation: Observation):
    width, height = (
        observation['world']['width'],
        observation['world']['height']
    )
    return width, height


def unit_dimensions():
    # max_binary_unit_value = dec2bin(MAX_UNIT_VALUE)
    # unit_dimensionality = len(max_binary_unit_value)
    return UNIT_DIMS # unit_dimensionality

def action_dimensions():
    action_dimensionality = len(ACTIONS)
    return action_dimensionality


def state_dimensions(observation: Observation):
    n_units = unit_dimensions()
    # n_coordinates = coordinates_dimensions(observation)
    n_width, n_height = map_dimensions(observation)
    n_cells = n_width * n_height
    state_dimensionality = n_units * n_cells #  + n_coordinates
    return state_dimensionality


"""
Map encoding: ICAART approach
"""


def empty_map(observation: Observation, with_ones: bool=False, length: int=0) -> Map:
    map_dims = map_dimensions(observation)
    if with_ones:
        return np.ones(map_dims, dtype=np.int8)
    if length > 0:
        width, height = map_dims
        return np.ones((width, height, length), dtype=np.int8)
    return np.zeros(map_dims, dtype=np.int8)


def cell_type_map(observation: Observation) -> Map:
    map = empty_map(observation, with_ones=True)
    for entity in observation['entities']:
        if entity['type'] == 'w':
            map[entity['x']][entity['y']] = 0
        if entity['type'] == 'o':
            map[entity['x']][entity['y']] = 0
        if entity['type'] == 'm':
            map[entity['x']][entity['y']] = -1
        if entity['type'] == 'b':
            map[entity['x']][entity['y']] = -1
    return map

def player_map(observation: Observation, current_agent_id: str) -> Map:
    map = empty_map(observation)
    for observed_agent_id, observed_agent_config in observation['agents'].items():
        for unit_id in observed_agent_config['unit_ids']:
            unit_config = observation['unit_state'][unit_id]
            if observed_agent_id == current_agent_id:
                map[unit_config['coordinates'][0]][unit_config['coordinates'][1]] = 1
    return map

def opponent_map(observation: Observation, current_agent_id: str) -> Map:
    map = empty_map(observation)
    for observed_agent_id, observed_agent_config in observation['agents'].items():
        for unit_id in observed_agent_config['unit_ids']:
            unit_config = observation['unit_state'][unit_id]
            if observed_agent_id != current_agent_id:
                map[unit_config['coordinates'][0]][unit_config['coordinates'][1]] = 1
    return map

def danger_zone_map(observation: Observation) -> Map:
    map = empty_map(observation)
    tick = observation['tick']
    for entity in observation['entities']:
        if entity['type'] == 'b':
            blast_diameter = entity['blast_diameter']
            time_left = entity['expires'] - tick
            i = 1
            while (i <= blast_diameter) and (entity['x'] + i < 15) and (entity['y'] + i < 15):
                map[entity['x'] + i][entity['y'] + i] = (blast_diameter - i + 1) / time_left
                i += 1
            i = 1
            while (i <= blast_diameter) and (entity['x'] - i >= 0) and (entity['y'] + i < 15):
                map[entity['x'] - i][entity['y'] + i] = (blast_diameter - i + 1) / time_left
                i += 1
            i = 1
            while (i <= blast_diameter) and (entity['x'] + i < 15) and (entity['y'] - i >= 0):
                map[entity['x'] + i][entity['y'] - i] = (blast_diameter - i + 1) / time_left
                i += 1
            i = 1
            while (i <= blast_diameter) and (entity['x'] - i >= 0) and (entity['y'] - i >= 0):
                map[entity['x'] - i][entity['y'] - i] = (blast_diameter - i + 1) / time_left
                i += 1
    return map


def powerup_map(observation: Observation) -> Map:
    map = empty_map(observation)
    for entity in observation['entities']:
        if entity['type'] == 'bp':
            map[entity['x']][entity['y']] = 1
        if entity['type'] == 'fp':
            map[entity['x']][entity['y']] = -1
    return map


"""
State encoding: ICAART approach
"""


def observation_to_state(observation: Observation, current_agent_id: str, current_unit_id: str):
    state = empty_map(observation, length=UNIT_DIMS)
    state[:, :, 0] = cell_type_map(observation)
    state[:, :, 1] = player_map(observation, current_agent_id)
    state[:, :, 2] = opponent_map(observation, current_agent_id)
    state[:, :, 3] = danger_zone_map(observation)
    state[:, :, 4] = powerup_map(observation)
    state = state.astype(np.float32)
    return torch.tensor(state).flatten()
