import argparse
import time
import torch

from components.environment.config import (
    FWD_MODEL_CONNECTION_DELAY,
    FWD_MODEL_CONNECTION_RETRIES,
)
from agent_base import Agent 

def main(path: str):
    print(f"Running reinforcement learning agent ({path})")
    for retry in range(FWD_MODEL_CONNECTION_RETRIES):
        try:
            Agent(model=torch.load(path))
        except Exception as e:
            print(f"Retrying to connect with {retry} attempt... Due to: {str(e)}")
            time.sleep(FWD_MODEL_CONNECTION_DELAY)
            continue
        break


parser = argparse.ArgumentParser(
                    prog='Agent runner',
                    description='Submission of trained Reinforcement Learning agent',
                    epilog='Happy coding!')
parser.add_argument('-p', '--path', help="Path to the agent you would like to submit (example: 'agent_ppo.pt')")

args = parser.parse_args()

main(args.path)
