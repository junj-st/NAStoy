"""
RL controller experiment on CIFAR-10.
"""

import json
import yaml
from pathlib import Path

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator
from nas.strategies.rl_controller import RLController
from nas import viz


def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'cifar10_rl.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running RL controller search on CIFAR-10")
    print(f"Config: {config}")

    # TODO: Implement experiment (input_size=3072 for CIFAR-10)

    print("RL search completed!")


if __name__ == '__main__':
    main()
