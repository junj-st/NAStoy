"""
RL-based controller for NAS.

Uses REINFORCE to train an LSTM controller that generates architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from nas.search_space import SearchSpace
from nas.evaluator import Evaluator


class LSTMController(nn.Module):
    """LSTM controller for architecture generation."""

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 1,
        vocab_sizes: List[int] = None
    ):
        """
        Args:
            hidden_size: LSTM hidden size
            n_layers: Number of LSTM layers
            vocab_sizes: List of vocabulary sizes for each token position
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_sizes = vocab_sizes or self._get_default_vocab_sizes()
        self.n_tokens = len(self.vocab_sizes)

        # Embedding layer (max vocab size across all positions)
        max_vocab = max(self.vocab_sizes)
        self.embedding = nn.Embedding(max_vocab, hidden_size)

        # LSTM cell
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        # Output heads - one linear layer per token position
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for vocab_size in self.vocab_sizes
        ])

    def _get_default_vocab_sizes(self) -> List[int]:
        """Get default vocabulary sizes for each token position.

        Returns:
            vocab_sizes: List of 13 vocabulary sizes
        """
        # n_layers: 0-3 (4 choices)
        # hidden_sizes: 0-3 (4 choices) × 4 positions
        # activations: 0-2 (3 choices) × 4 positions
        # dropouts: 0-3 (4 choices) × 4 positions
        return [4] + [4]*4 + [3]*4 + [4]*4

    def forward(self, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate architecture tokens.

        Args:
            x: Optional input tokens (for teacher forcing)

        Returns:
            logits: Token logits [batch_size, n_tokens, vocab_size]
            probs: Token probabilities
        """
        # TODO: Implement forward pass
        # Autoregressive generation: each token depends on previous tokens
        raise NotImplementedError

    def sample(self, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample architectures from controller.

        Args:
            n_samples: Number of architectures to sample

        Returns:
            tokens: Sampled token sequences [n_samples, 13]
            log_probs: Log probabilities of sampled tokens
            entropy: Entropy of token distributions
        """
        self.eval()

        tokens = []
        log_probs = []
        entropies = []

        # Initialize hidden state
        h = torch.zeros(self.n_layers, n_samples, self.hidden_size)
        c = torch.zeros(self.n_layers, n_samples, self.hidden_size)

        # Start token (zeros)
        input_token = torch.zeros(n_samples, 1, self.hidden_size)

        # Autoregressive generation
        for i in range(self.n_tokens):
            # LSTM forward
            output, (h, c) = self.lstm(input_token, (h, c))

            # Get logits for this position
            logits = self.output_heads[i](output.squeeze(1))  # [n_samples, vocab_size]

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            token = dist.sample()  # [n_samples]

            # Track log probability and entropy
            log_prob = dist.log_prob(token)
            entropy = dist.entropy()

            tokens.append(token)
            log_probs.append(log_prob)
            entropies.append(entropy)

            # Embed sampled token as input for next step
            input_token = self.embedding(token).unsqueeze(1)  # [n_samples, 1, hidden_size]

        # Stack results
        tokens = torch.stack(tokens, dim=1)  # [n_samples, n_tokens]
        log_probs = torch.stack(log_probs, dim=1)  # [n_samples, n_tokens]
        entropies = torch.stack(entropies, dim=1)  # [n_samples, n_tokens]

        return tokens, log_probs.sum(dim=1), entropies.mean(dim=1)


class RLController:
    """RL-based architecture search using REINFORCE."""

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        n_episodes: int = 200,
        batch_size: int = 10,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
        entropy_coeff: float = 0.01,
        entropy_decay: float = 0.995,
        reward_lambda: float = 0.001,
        seed: Optional[int] = None
    ):
        """
        Args:
            search_space: Architecture search space
            evaluator: Architecture evaluator
            n_episodes: Number of training episodes
            batch_size: Number of architectures per episode
            learning_rate: Learning rate for controller
            hidden_size: LSTM hidden size
            entropy_coeff: Entropy regularization coefficient
            entropy_decay: Decay rate for entropy coefficient
            reward_lambda: Penalty coefficient for model size
            seed: Random seed
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.reward_lambda = reward_lambda

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize controller and optimizer
        self.controller = LSTMController(hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=learning_rate)
        self.baseline = None  # EMA baseline for variance reduction

        self.history = []
        self.all_results = []  # Track all evaluated architectures

    def search(self) -> List[Dict]:
        """Run RL-based search.

        Returns:
            results: Best architectures found
        """
        print(f"Starting RL-based search...")
        print(f"  Episodes: {self.n_episodes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  Entropy coeff: {self.entropy_coeff}")
        print(f"  Reward lambda: {self.reward_lambda}\n")

        for episode in tqdm(range(1, self.n_episodes + 1), desc="RL Training"):
            # Sample architectures from controller
            tokens, log_probs, entropy = self.controller.sample(self.batch_size)

            # Evaluate each architecture
            episode_results = []
            for i in range(self.batch_size):
                # Decode tokens to config
                config = self.search_space.decode(tokens[i].tolist())

                # Build and evaluate
                model = self.search_space.build_model(config)
                result = self.evaluator.evaluate(model, config)

                episode_results.append(result)
                self.all_results.append(result)

            # Compute rewards
            rewards = torch.tensor([self.compute_reward(r) for r in episode_results])

            # Update controller with REINFORCE
            loss, policy_loss, entropy_loss = self.update_controller(
                log_probs, rewards, entropy
            )

            # Track episode stats
            episode_stats = {
                'episode': episode,
                'mean_reward': rewards.mean().item(),
                'best_reward': rewards.max().item(),
                'mean_acc': np.mean([r['val_acc'] for r in episode_results]),
                'best_acc': max([r['val_acc'] for r in episode_results]),
                'entropy': entropy.mean().item(),
                'loss': loss,
                'baseline': self.baseline
            }
            self.history.append(episode_stats)

            # Decay entropy coefficient
            self.entropy_coeff *= self.entropy_decay

            # Log progress
            if episode % 10 == 0:
                print(f"\nEpisode {episode}: "
                      f"Reward={rewards.mean().item():.4f}, "
                      f"Acc={episode_stats['mean_acc']:.4f}, "
                      f"Entropy={episode_stats['entropy']:.4f}, "
                      f"EntCoeff={self.entropy_coeff:.6f}")

        print(f"\nRL search complete!")

        # Sort all results by accuracy
        self.all_results.sort(key=lambda x: x['val_acc'], reverse=True)

        print(f"Best architecture: {self.all_results[0]['val_acc']:.4f} accuracy")

        return self.all_results

    def compute_reward(self, result: Dict) -> float:
        """Compute reward from evaluation result.

        Args:
            result: Evaluation result

        Returns:
            reward: Shaped reward (accuracy - size penalty)
        """
        # Reward = validation accuracy - size penalty
        val_acc = result['val_acc']
        n_params = result['n_params']

        # Penalize large models
        size_penalty = self.reward_lambda * np.log(n_params)

        reward = val_acc - size_penalty
        return reward

    def update_controller(self, log_probs: torch.Tensor,
                         rewards: torch.Tensor, entropy: torch.Tensor):
        """Update controller with REINFORCE.

        Args:
            log_probs: Log probabilities of tokens
            rewards: Rewards for each architecture
            entropy: Entropy of token distributions
        """
        # Update baseline (EMA)
        if self.baseline is None:
            self.baseline = rewards.mean().item()
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * rewards.mean().item()

        # REINFORCE loss with baseline
        advantages = rewards - self.baseline
        policy_loss = -(advantages * log_probs).mean()

        # Entropy regularization (encourage exploration)
        entropy_loss = -self.entropy_coeff * entropy.mean()

        # Total loss
        loss = policy_loss + entropy_loss

        # Update controller
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), policy_loss.item(), entropy_loss.item()

    def get_best_architecture(self) -> Dict:
        """Get best architecture found.

        Returns:
            best_result: Best evaluation result
        """
        if not self.all_results:
            raise ValueError("No architectures evaluated yet")
        return self.all_results[0]
