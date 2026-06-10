"""
RL controller experiment on MNIST.
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
    config_path = Path(__file__).parent.parent / 'configs' / 'mnist_rl.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running RL controller search on MNIST")
    print(f"Config: {config}")

    # TODO: Implement experiment
    # 1. Initialize search space
    # 2. Initialize evaluator
    # 3. Initialize RL controller strategy
    # 4. Run search
    # 5. Save results
    # 6. Generate visualizations (trajectory + RL metrics)
    # 7. Print best architecture

    print("RL search completed!")


if __name__ == '__main__':
    main()
