"""Reference implementation for [14.4] Maze Transformers and Latent Planning.

The model is trained from scratch on deterministic mazes with exact breadth-first
search labels. It predicts one shortest-path action from a serialized maze state;
autoregressive planning repeatedly feeds the resulting state back to the model.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch as t
from torch import Tensor, nn
from torch.nn import functional as F


MAX_MAZE_SIZE = 7
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")

PAD = 0
BOS = 1
SIZE_5 = 2
SIZE_6 = 3
SIZE_7 = 4
OPEN = 5
WALL = 6
CURRENT = 7
GOAL = 8
CURRENT_GOAL = 9
QUERY = 10
VOCAB_SIZE = 11
SEQUENCE_LENGTH = 2 + MAX_MAZE_SIZE**2 + 1
QUERY_POSITION = SEQUENCE_LENGTH - 1


@dataclass(frozen=True)
class Maze:
    walls: Tensor
    start: tuple[int, int]
    goal: tuple[int, int]
    path: tuple[tuple[int, int], ...]

    @property
    def size(self) -> int:
        return int(self.walls.shape[0])

    @property
    def distance(self) -> int:
        return len(self.path) - 1


@dataclass(frozen=True)
class StateExample:
    maze_id: int
    current: tuple[int, int]
    goal: tuple[int, int]
    action: int
    distance: int
    legal_degree: int
    template_is_shortest: bool


@dataclass(frozen=True)
class MazeDataset:
    tokens: Tensor
    actions: Tensor
    distances: Tensor
    legal_degrees: Tensor
    states: tuple[StateExample, ...]


@dataclass(frozen=True)
class Rollout:
    cells: tuple[tuple[int, int], ...]
    actions: tuple[int, ...]
    legal: bool
    reached_goal: bool
    optimal: bool


@dataclass(frozen=True)
class RolloutMetrics:
    legal_path_rate: float
    goal_reach_rate: float
    optimal_path_rate: float
    mean_excess_steps: float
    illegal_move_rate: float


@dataclass(frozen=True)
class TrainingResult:
    train_loss: tuple[float, ...]
    heldout_action_accuracy: tuple[float, ...]
    checkpoint_epochs: tuple[int, ...]
    checkpoints: tuple[dict[str, Tensor], ...]


@dataclass(frozen=True)
class ProbeResult:
    distance_accuracy: tuple[float, ...]
    legal_degree_accuracy: tuple[float, ...]
    action_accuracy: tuple[float, ...]
    random_label_distance_accuracy: tuple[float, ...]
    manhattan_distance_accuracy: float


@dataclass(frozen=True)
class PatchingResult:
    heatmap: Tensor
    targeted_effect: float
    targeted_positive_fraction: float
    wrong_position_effect: float
    matched_norm_random_effect: float
    unrelated_route_effect: float
    same_action_other_maze_effect: float
    same_action_targeted_effect: float
    same_action_pair_count: int
    pair_count: int


@dataclass(frozen=True)
class ExperimentArtifacts:
    model: "TinyMazeTransformer"
    train_mazes: tuple[Maze, ...]
    heldout_mazes: tuple[Maze, ...]
    density_ood_mazes: tuple[Maze, ...]
    size_ood_mazes: tuple[Maze, ...]
    training: TrainingResult
    probes: ProbeResult
    patching: PatchingResult
    model_rollouts: tuple[Rollout, ...]
    bfs_paths: tuple[tuple[tuple[int, int], ...], ...]
    metrics: dict[str, float | int | bool]


def legal_neighbors(walls: Tensor, cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Return open in-bounds neighbors in deterministic action order."""

    row, column = cell
    size = int(walls.shape[0])
    neighbors: list[tuple[int, int]] = []
    for delta_row, delta_column in ACTION_DELTAS:
        next_row, next_column = row + delta_row, column + delta_column
        if 0 <= next_row < size and 0 <= next_column < size:
            if not bool(walls[next_row, next_column]):
                neighbors.append((next_row, next_column))
    return neighbors


def bfs_shortest_path(
    walls: Tensor,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    """Compute an exact shortest path with breadth-first search."""

    if walls.ndim != 2 or walls.shape[0] != walls.shape[1]:
        raise ValueError("walls must be a square rank-2 tensor")
    size = int(walls.shape[0])
    for name, (row, column) in (("start", start), ("goal", goal)):
        if not (0 <= row < size and 0 <= column < size):
            raise ValueError(f"{name} lies outside the maze")
        if bool(walls[row, column]):
            raise ValueError(f"{name} must be on an open cell")

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path: list[tuple[int, int]] = []
            while cell is not None:
                path.append(cell)
                cell = parents[cell]
            return tuple(reversed(path))
        for neighbor in legal_neighbors(walls, cell):
            if neighbor not in parents:
                parents[neighbor] = cell
                queue.append(neighbor)
    return None


def distance_map(walls: Tensor, goal: tuple[int, int]) -> Tensor:
    """Return exact BFS distance to the goal, with -1 for unreachable cells."""

    distances = t.full_like(walls, -1, dtype=t.long)
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for neighbor in legal_neighbors(walls, cell):
            if distances[neighbor] < 0:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def action_between(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Return the action index for one adjacent transition."""

    delta = (second[0] - first[0], second[1] - first[1])
    if delta not in ACTION_DELTAS:
        raise ValueError(f"cells must be cardinal neighbors; got delta={delta}")
    return ACTION_DELTAS.index(delta)


def _maze_key(maze: Maze) -> tuple[Any, ...]:
    return (
        maze.size,
        bytes(maze.walls.to(t.uint8).flatten().tolist()),
    )


def generate_maze(
    seed: int,
    *,
    size: int = 5,
    obstacle_density: float = 0.16,
    min_distance: int = 4,
    require_detour: bool = False,
) -> Maze:
    """Generate a deterministic solvable maze with exact BFS ground truth."""

    if not 3 <= size <= MAX_MAZE_SIZE:
        raise ValueError(f"size must lie in [3, {MAX_MAZE_SIZE}]")
    if not 0.0 <= obstacle_density < 0.5:
        raise ValueError("obstacle_density must lie in [0, 0.5)")
    rng = random.Random(seed)
    cells = [(row, column) for row in range(size) for column in range(size)]
    for _ in range(2_000):
        start, goal = rng.sample(cells, 2)
        walls = t.zeros(size, size, dtype=t.bool)
        for row, column in cells:
            walls[row, column] = rng.random() < obstacle_density
        walls[start] = False
        walls[goal] = False
        path = bfs_shortest_path(walls, start, goal)
        if path is None or len(path) - 1 < min_distance:
            continue
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if require_detour and len(path) - 1 <= manhattan:
            continue
        return Maze(walls=walls, start=start, goal=goal, path=path)
    raise RuntimeError("failed to sample a maze satisfying the requested constraints")


def generate_maze_collection(
    count: int,
    *,
    seed: int,
    size: int = 5,
    obstacle_density: float = 0.16,
    require_detour_fraction: float = 0.25,
    excluded_keys: set[tuple[Any, ...]] | None = None,
) -> tuple[Maze, ...]:
    """Generate unique mazes, optionally disjoint from another split."""

    excluded = set() if excluded_keys is None else set(excluded_keys)
    mazes: list[Maze] = []
    candidate = seed
    while len(mazes) < count:
        require_detour = len(mazes) < round(count * require_detour_fraction)
        try:
            maze = generate_maze(
                candidate,
                size=size,
                obstacle_density=obstacle_density,
                min_distance=4,
                require_detour=require_detour,
            )
        except RuntimeError:
            candidate += 1
            continue
        key = _maze_key(maze)
        if key not in excluded:
            mazes.append(maze)
            excluded.add(key)
        candidate += 1
    return tuple(mazes)


def tokenize_maze_state(
    maze: Maze,
    current: tuple[int, int],
    *,
    max_size: int = MAX_MAZE_SIZE,
) -> Tensor:
    """Serialize a maze state as a fixed-length causal token sequence."""

    if not (0 <= current[0] < maze.size and 0 <= current[1] < maze.size):
        raise ValueError("current lies outside the maze")
    if bool(maze.walls[current]):
        raise ValueError("current must be an open cell")
    size_token = {5: SIZE_5, 6: SIZE_6, 7: SIZE_7}.get(maze.size)
    if size_token is None:
        raise ValueError("this course vocabulary supports maze sizes 5, 6, and 7")
    grid_tokens: list[int] = []
    for row in range(max_size):
        for column in range(max_size):
            if row >= maze.size or column >= maze.size or bool(maze.walls[row, column]):
                token = WALL
            else:
                token = OPEN
            if (row, column) == maze.goal:
                token = GOAL
            if (row, column) == current:
                token = CURRENT_GOAL if current == maze.goal else CURRENT
            grid_tokens.append(token)
    return t.tensor([BOS, size_token, *grid_tokens, QUERY], dtype=t.long)


def template_action(current: tuple[int, int], goal: tuple[int, int]) -> int:
    """Axis-first action which ignores walls."""

    if current[1] < goal[1]:
        return 1
    if current[1] > goal[1]:
        return 3
    if current[0] < goal[0]:
        return 2
    return 0


def build_state_dataset(mazes: tuple[Maze, ...]) -> MazeDataset:
    """Create an exact shortest-action example at every reachable non-goal cell."""

    tokens: list[Tensor] = []
    actions: list[int] = []
    distances: list[int] = []
    legal_degrees: list[int] = []
    states: list[StateExample] = []
    for maze_id, maze in enumerate(mazes):
        distance_grid = distance_map(maze.walls, maze.goal)
        for row in range(maze.size):
            for column in range(maze.size):
                current = (row, column)
                distance = int(distance_grid[current])
                if distance <= 0:
                    continue
                downhill = [
                    neighbor
                    for neighbor in legal_neighbors(maze.walls, current)
                    if int(distance_grid[neighbor]) == distance - 1
                ]
                if not downhill:
                    raise AssertionError("reachable non-goal cell has no BFS-downhill neighbor")
                next_cell = downhill[0]
                action = action_between(current, next_cell)
                legal_degree = len(legal_neighbors(maze.walls, current))
                delta_row, delta_column = ACTION_DELTAS[template_action(current, maze.goal)]
                template_next = (row + delta_row, column + delta_column)
                template_is_shortest = (
                    template_next in legal_neighbors(maze.walls, current)
                    and int(distance_grid[template_next]) == distance - 1
                )
                tokens.append(tokenize_maze_state(maze, current))
                actions.append(action)
                distances.append(distance)
                legal_degrees.append(legal_degree)
                states.append(
                    StateExample(
                        maze_id=maze_id,
                        current=current,
                        goal=maze.goal,
                        action=action,
                        distance=distance,
                        legal_degree=legal_degree,
                        template_is_shortest=template_is_shortest,
                    )
                )
    return MazeDataset(
        tokens=t.stack(tokens),
        actions=t.tensor(actions),
        distances=t.tensor(distances),
        legal_degrees=t.tensor(legal_degrees) - 1,
        states=tuple(states),
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, sequence, width = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal_mask = t.triu(
            t.ones(sequence, sequence, dtype=t.bool, device=x.device), diagonal=1
        )
        attention = scores.masked_fill(causal_mask, -t.inf).softmax(dim=-1)
        head_values = attention @ value
        joined = head_values.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(joined), attention, head_values


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        residual_pre = x
        attention_output, pattern, head_values = self.attention(self.norm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.norm_2(x))
        x = x + mlp_output
        return x, {
            "resid_pre": residual_pre,
            "attn_out": attention_output,
            "mlp_out": mlp_output,
            "resid_post": x,
            "pattern": pattern,
            "head_values": head_values,
        }


class TinyMazeTransformer(nn.Module):
    """Decoder-only transformer with explicit layer/position activation cache."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_mlp: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Parameter(t.randn(SEQUENCE_LENGTH, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, len(ACTION_NAMES))

    def forward(
        self,
        tokens: Tensor,
        *,
        return_cache: bool = False,
        residual_patch: tuple[int, int, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        if tokens.shape[-1] != SEQUENCE_LENGTH:
            raise ValueError(f"expected sequence length {SEQUENCE_LENGTH}")
        x = self.token_embedding(tokens) + self.position_embedding[None]
        cache: dict[str, Any] = {"embed": x, "blocks": []}
        for layer, block in enumerate(self.blocks):
            x, block_cache = block(x)
            if residual_patch is not None and residual_patch[0] == layer:
                _, position, replacement = residual_patch
                x = x.clone()
                x[:, position] = replacement
                block_cache["resid_post"] = x
            cache["blocks"].append(block_cache)
        logits = self.action_head(self.final_norm(x[:, QUERY_POSITION]))
        if return_cache:
            cache["final"] = x
            return logits, cache
        return logits


@t.no_grad()
def action_accuracy(model: TinyMazeTransformer, dataset: MazeDataset, batch_size: int = 256) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(dataset.tokens), batch_size):
        tokens = dataset.tokens[start : start + batch_size].to(next(model.parameters()).device)
        labels = dataset.actions[start : start + batch_size].to(tokens.device)
        correct += int(model(tokens).argmax(-1).eq(labels).sum().item())
    return correct / len(dataset.tokens)


def train_maze_transformer(
    train_data: MazeDataset,
    heldout_data: MazeDataset,
    *,
    device: str | t.device = "cpu",
    seed: int = 0,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    d_model: int = 64,
    n_layers: int = 3,
) -> tuple[TinyMazeTransformer, TrainingResult]:
    """Train a real maze policy and retain immutable state-dict checkpoints."""

    device = t.device(device)
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the CUDA process.")
        t.use_deterministic_algorithms(True)
        t.set_float32_matmul_precision("highest")
    t.manual_seed(seed)
    model = TinyMazeTransformer(d_model=d_model, n_layers=n_layers).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = t.Generator().manual_seed(seed + 1)
    train_loss: list[float] = []
    heldout_accuracy: list[float] = []
    checkpoint_epochs: list[int] = []
    checkpoints: list[dict[str, Tensor]] = []

    for epoch in range(epochs):
        model.train()
        ordering = t.randperm(len(train_data.tokens), generator=generator)
        losses: list[float] = []
        for indices in ordering.split(batch_size):
            tokens = train_data.tokens[indices].to(device)
            labels = train_data.actions[indices].to(device)
            loss = F.cross_entropy(model(tokens), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        train_loss.append(sum(losses) / len(losses))
        heldout_accuracy.append(action_accuracy(model, heldout_data))
        if epoch in {0, epochs // 2, epochs - 1}:
            checkpoint_epochs.append(epoch + 1)
            checkpoints.append(
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            )
    return model, TrainingResult(
        train_loss=tuple(train_loss),
        heldout_action_accuracy=tuple(heldout_accuracy),
        checkpoint_epochs=tuple(checkpoint_epochs),
        checkpoints=tuple(checkpoints),
    )


def apply_action(
    walls: Tensor,
    current: tuple[int, int],
    action: int,
) -> tuple[int, int] | None:
    if not 0 <= action < len(ACTION_DELTAS):
        raise ValueError("action must be one of 0, 1, 2, 3")
    delta_row, delta_column = ACTION_DELTAS[action]
    candidate = (current[0] + delta_row, current[1] + delta_column)
    return candidate if candidate in legal_neighbors(walls, current) else None


@t.no_grad()
def rollout_model(
    model: TinyMazeTransformer,
    maze: Maze,
    *,
    max_steps: int | None = None,
    constrain_legal: bool = False,
) -> Rollout:
    """Autoregress by repeatedly querying the model on its updated current cell."""

    model.eval()
    device = next(model.parameters()).device
    max_steps = 2 * maze.size**2 if max_steps is None else max_steps
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    cells = [maze.start]
    actions: list[int] = []
    seen = {maze.start}
    legal = True
    for _ in range(max_steps):
        current = cells[-1]
        if current == maze.goal:
            break
        tokens = tokenize_maze_state(maze, current).unsqueeze(0).to(device)
        logits = model(tokens)[0]
        if constrain_legal:
            legal_actions = [
                action_index
                for action_index in range(len(ACTION_NAMES))
                if apply_action(maze.walls, current, action_index) is not None
            ]
            action = max(legal_actions, key=lambda index: float(logits[index]))
        else:
            action = int(logits.argmax().item())
        actions.append(action)
        next_cell = apply_action(maze.walls, current, action)
        if next_cell is None:
            legal = False
            break
        cells.append(next_cell)
        if next_cell in seen and next_cell != maze.goal:
            break
        seen.add(next_cell)
    reached = cells[-1] == maze.goal
    return Rollout(
        cells=tuple(cells),
        actions=tuple(actions),
        legal=legal,
        reached_goal=reached,
        optimal=reached and len(actions) == maze.distance,
    )


def template_rollout(maze: Maze) -> Rollout:
    """Axis-first Manhattan policy which ignores walls until it collides."""

    current = maze.start
    cells = [current]
    actions: list[int] = []
    seen = {current}
    legal = True
    for _ in range(2 * maze.size**2):
        if current == maze.goal:
            break
        if current[1] < maze.goal[1]:
            action = 1
        elif current[1] > maze.goal[1]:
            action = 3
        elif current[0] < maze.goal[0]:
            action = 2
        else:
            action = 0
        actions.append(action)
        next_cell = apply_action(maze.walls, current, action)
        if next_cell is None:
            legal = False
            break
        current = next_cell
        cells.append(current)
        if current in seen and current != maze.goal:
            break
        seen.add(current)
    reached = current == maze.goal
    return Rollout(tuple(cells), tuple(actions), legal, reached, reached and len(actions) == maze.distance)


def score_rollouts(rollouts: tuple[Rollout, ...], mazes: tuple[Maze, ...]) -> RolloutMetrics:
    """Recompute legality, reach, and optimality from actions and actual walls.

    Stored boolean flags are convenient display metadata, not scoring evidence.
    Cell traces must agree with the independently replayed actions.
    """
    if len(rollouts) != len(mazes) or not rollouts:
        raise ValueError("rollouts and mazes must be nonempty and aligned")
    legal_flags, reached_flags, optimal_flags, excess = [], [], [], []
    for rollout, maze in zip(rollouts, mazes):
        replay = [maze.start]
        legal = True
        for step, action in enumerate(rollout.actions):
            if replay[-1] == maze.goal:
                raise ValueError("a rollout must stop when it reaches the goal")
            next_cell = apply_action(maze.walls, replay[-1], action)
            if next_cell is None:
                legal = False
                if step != len(rollout.actions) - 1:
                    raise ValueError("an illegal move must terminate the rollout")
                break
            replay.append(next_cell)
        if tuple(replay) != rollout.cells:
            raise ValueError("cell trace does not match replayed actions")
        shortest = bfs_shortest_path(maze.walls, maze.start, maze.goal)
        reached = legal and replay[-1] == maze.goal
        distance = len(shortest) - 1 if shortest is not None else math.inf
        optimal = reached and len(rollout.actions) == distance
        legal_flags.append(legal)
        reached_flags.append(reached)
        optimal_flags.append(optimal)
        if reached:
            excess.append(len(rollout.actions) - distance)
    return RolloutMetrics(
        legal_path_rate=sum(legal_flags) / len(rollouts),
        goal_reach_rate=sum(reached_flags) / len(rollouts),
        optimal_path_rate=sum(optimal_flags) / len(rollouts),
        mean_excess_steps=sum(excess) / max(1, len(excess)),
        illegal_move_rate=sum(not legal for legal in legal_flags) / len(rollouts),
    )


@t.no_grad()
def extract_layer_features(
    model: TinyMazeTransformer,
    dataset: MazeDataset,
    *,
    max_examples: int = 1_200,
    batch_size: int = 256,
) -> Tensor:
    """Return query-position features as [layers+1, examples, d_model]."""

    model.eval()
    device = next(model.parameters()).device
    layer_chunks: list[list[Tensor]] = [[] for _ in range(model.n_layers + 1)]
    tokens = dataset.tokens[:max_examples]
    for batch in tokens.split(batch_size):
        _, cache = model(batch.to(device), return_cache=True)
        layer_chunks[0].append(cache["embed"][:, QUERY_POSITION].cpu())
        for layer, block_cache in enumerate(cache["blocks"]):
            layer_chunks[layer + 1].append(block_cache["resid_post"][:, QUERY_POSITION].cpu())
    return t.stack([t.cat(chunks) for chunks in layer_chunks])


def ridge_probe_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    n_classes: int,
    ridge: float = 1e-2,
) -> float:
    """Closed-form standardized multiclass ridge probe."""

    mean = train_features.float().mean(0, keepdim=True)
    scale = train_features.float().std(0, keepdim=True).clamp_min(1e-5)
    train = (train_features.float() - mean) / scale
    test = (test_features.float() - mean) / scale
    train = t.cat((train, t.ones(len(train), 1)), dim=1)
    test = t.cat((test, t.ones(len(test), 1)), dim=1)
    targets = F.one_hot(train_labels, n_classes).float()
    identity = t.eye(train.shape[1])
    identity[-1, -1] = 0
    weights = t.linalg.solve(train.T @ train + ridge * identity, train.T @ targets)
    return (test @ weights).argmax(-1).eq(test_labels).float().mean().item()


def layerwise_probe_metrics(
    model: TinyMazeTransformer,
    train_data: MazeDataset,
    heldout_data: MazeDataset,
    *,
    seed: int = 0,
) -> ProbeResult:
    train_features = extract_layer_features(model, train_data)
    heldout_features = extract_layer_features(model, heldout_data)
    train_count = train_features.shape[1]
    heldout_count = heldout_features.shape[1]
    distance_boundaries = t.tensor([2, 4])
    train_distance = t.bucketize(
        train_data.distances[:train_count], distance_boundaries
    )
    heldout_distance = t.bucketize(
        heldout_data.distances[:heldout_count], distance_boundaries
    )
    train_legal_degree = train_data.legal_degrees[:train_count]
    heldout_legal_degree = heldout_data.legal_degrees[:heldout_count]
    train_action = train_data.actions[:train_count]
    heldout_action = heldout_data.actions[:heldout_count]
    generator = t.Generator().manual_seed(seed)
    shuffled_train = train_distance[t.randperm(train_count, generator=generator)]
    manhattan_distances = t.tensor([
        abs(state.current[0] - state.goal[0]) + abs(state.current[1] - state.goal[1])
        for state in heldout_data.states[:heldout_count]
    ])
    manhattan_accuracy = float(
        t.bucketize(manhattan_distances, distance_boundaries).eq(heldout_distance).float().mean()
    )

    distance_scores: list[float] = []
    legal_degree_scores: list[float] = []
    action_scores: list[float] = []
    random_scores: list[float] = []
    for train_layer, test_layer in zip(train_features, heldout_features):
        distance_scores.append(
            ridge_probe_accuracy(
                train_layer, train_distance, test_layer, heldout_distance, n_classes=3
            )
        )
        legal_degree_scores.append(
            ridge_probe_accuracy(
                train_layer, train_legal_degree, test_layer, heldout_legal_degree, n_classes=4
            )
        )
        action_scores.append(
            ridge_probe_accuracy(train_layer, train_action, test_layer, heldout_action, n_classes=4)
        )
        random_scores.append(
            ridge_probe_accuracy(
                train_layer, shuffled_train, test_layer, heldout_distance, n_classes=3
            )
        )
    return ProbeResult(
        distance_accuracy=tuple(distance_scores),
        legal_degree_accuracy=tuple(legal_degree_scores),
        action_accuracy=tuple(action_scores),
        random_label_distance_accuracy=tuple(random_scores),
        manhattan_distance_accuracy=manhattan_accuracy,
    )


def shuffled_maze_accuracy(
    model: TinyMazeTransformer,
    dataset: MazeDataset,
    *,
    seed: int = 0,
) -> float:
    """Shuffle wall layouts while retaining each example's current/goal markers."""

    generator = t.Generator().manual_seed(seed)
    source_order = t.randperm(len(dataset.tokens), generator=generator)
    corrupted = dataset.tokens.clone()
    grid_slice = slice(2, 2 + MAX_MAZE_SIZE**2)
    source_grid = dataset.tokens[source_order, grid_slice].clone()
    source_grid[(source_grid == CURRENT) | (source_grid == GOAL) | (source_grid == CURRENT_GOAL)] = OPEN
    original_grid = dataset.tokens[:, grid_slice]
    source_grid[original_grid == CURRENT] = CURRENT
    source_grid[original_grid == GOAL] = GOAL
    source_grid[original_grid == CURRENT_GOAL] = CURRENT_GOAL
    corrupted[:, grid_slice] = source_grid
    corrupt_dataset = MazeDataset(
        tokens=corrupted,
        actions=dataset.actions,
        distances=dataset.distances,
        legal_degrees=dataset.legal_degrees,
        states=dataset.states,
    )
    return action_accuracy(model, corrupt_dataset)




@t.no_grad()
def detour_action_accuracy(
    model: TinyMazeTransformer,
    dataset: MazeDataset,
    *,
    shuffled_mazes: bool = False,
    seed: int = 0,
) -> float:
    """Evaluate states where the wall-blind move is blocked or not BFS-downhill.

    A different tie break between equally short moves is not a detour.
    """

    indices = t.tensor(
        [
            index
            for index, state in enumerate(dataset.states)
            if not state.template_is_shortest
        ]
    )
    if indices.numel() == 0:
        raise ValueError("dataset has no states where the template is not shortest")
    tokens = dataset.tokens.clone()
    if shuffled_mazes:
        generator = t.Generator().manual_seed(seed)
        order = t.randperm(len(tokens), generator=generator)
        grid_slice = slice(2, 2 + MAX_MAZE_SIZE**2)
        source = tokens[order, grid_slice].clone()
        source[(source == CURRENT) | (source == GOAL) | (source == CURRENT_GOAL)] = OPEN
        original = tokens[:, grid_slice]
        source[original == CURRENT] = CURRENT
        source[original == GOAL] = GOAL
        source[original == CURRENT_GOAL] = CURRENT_GOAL
        tokens[:, grid_slice] = source
    subset = MazeDataset(
        tokens=tokens[indices],
        actions=dataset.actions[indices],
        distances=dataset.distances[indices],
        legal_degrees=dataset.legal_degrees[indices],
        states=tuple(dataset.states[index] for index in indices.tolist()),
    )
    return action_accuracy(model, subset)


def _state_tokens(maze: Maze, cell: tuple[int, int], device: t.device) -> Tensor:
    return tokenize_maze_state(maze, cell).unsqueeze(0).to(device)


def _action_margin(logits: Tensor, target_action: int, clean_action: int) -> float:
    return float((logits[0, target_action] - logits[0, clean_action]).item())


@t.no_grad()
def residual_patching_heatmap(
    model: TinyMazeTransformer,
    maze: Maze,
    clean_index: int,
    corrupt_index: int,
) -> Tensor:
    """Patch every layer/position and normalize recovery of the corrupt action margin."""

    model.eval()
    device = next(model.parameters()).device
    clean_action = action_between(maze.path[clean_index], maze.path[clean_index + 1])
    corrupt_action = action_between(maze.path[corrupt_index], maze.path[corrupt_index + 1])
    if clean_action == corrupt_action:
        raise ValueError("clean and corrupt route states must require different actions")
    clean_tokens = _state_tokens(maze, maze.path[clean_index], device)
    corrupt_tokens = _state_tokens(maze, maze.path[corrupt_index], device)
    clean_logits, _ = model(clean_tokens, return_cache=True)
    corrupt_logits, corrupt_cache = model(corrupt_tokens, return_cache=True)
    clean_margin = _action_margin(clean_logits, corrupt_action, clean_action)
    corrupt_margin = _action_margin(corrupt_logits, corrupt_action, clean_action)
    denominator = corrupt_margin - clean_margin
    if abs(denominator) < 1e-5:
        raise ValueError("model does not distinguish the selected route states")
    heatmap = t.zeros(model.n_layers, SEQUENCE_LENGTH)
    for layer in range(model.n_layers):
        for position in range(SEQUENCE_LENGTH):
            replacement = corrupt_cache["blocks"][layer]["resid_post"][:, position]
            patched = model(
                clean_tokens,
                residual_patch=(layer, position, replacement),
            )
            patched_margin = _action_margin(patched, corrupt_action, clean_action)
            heatmap[layer, position] = (patched_margin - clean_margin) / denominator
    return heatmap


def _counterfactual_pairs(
    model: TinyMazeTransformer,
    mazes: tuple[Maze, ...],
    *,
    max_pairs: int = 24,
) -> list[tuple[Maze, int, int]]:
    pairs: list[tuple[Maze, int, int]] = []
    device = next(model.parameters()).device
    with t.no_grad():
        for maze in mazes:
            route_actions = [action_between(a, b) for a, b in zip(maze.path[:-1], maze.path[1:])]
            for first in range(len(route_actions)):
                for second in range(first + 1, len(route_actions)):
                    if route_actions[first] == route_actions[second]:
                        continue
                    first_prediction = int(model(_state_tokens(maze, maze.path[first], device)).argmax(-1))
                    second_prediction = int(model(_state_tokens(maze, maze.path[second], device)).argmax(-1))
                    if (
                        first_prediction == route_actions[first]
                        and second_prediction == route_actions[second]
                    ):
                        pairs.append((maze, first, second))
                        break
                if pairs and pairs[-1][0] is maze:
                    break
            if len(pairs) >= max_pairs:
                break
    return pairs


@t.no_grad()
def route_state_intervention_metrics(
    model: TinyMazeTransformer,
    mazes: tuple[Maze, ...],
    *,
    seed: int = 0,
) -> PatchingResult:
    """Compare targeted query patches with three matched replacement controls."""

    pairs = _counterfactual_pairs(model, mazes)
    if len(pairs) < 4:
        raise RuntimeError("not enough correctly solved counterfactual route-state pairs")
    device = next(model.parameters()).device
    layer = max(0, model.n_layers - 2)
    generator = t.Generator(device=device).manual_seed(seed)
    targeted: list[float] = []
    wrong_position: list[float] = []
    matched_random: list[float] = []
    unrelated: list[float] = []
    same_action: list[float] = []
    same_action_targeted: list[float] = []

    for pair_index, (maze, clean_index, corrupt_index) in enumerate(pairs):
        clean_action = action_between(maze.path[clean_index], maze.path[clean_index + 1])
        corrupt_action = action_between(maze.path[corrupt_index], maze.path[corrupt_index + 1])
        clean_tokens = _state_tokens(maze, maze.path[clean_index], device)
        corrupt_tokens = _state_tokens(maze, maze.path[corrupt_index], device)
        clean_logits, clean_cache = model(clean_tokens, return_cache=True)
        corrupt_logits, corrupt_cache = model(corrupt_tokens, return_cache=True)
        clean_margin = _action_margin(clean_logits, corrupt_action, clean_action)
        corrupt_margin = _action_margin(corrupt_logits, corrupt_action, clean_action)
        denominator = max(1e-5, corrupt_margin - clean_margin)
        clean_query = clean_cache["blocks"][layer]["resid_post"][:, QUERY_POSITION]
        corrupt_query = corrupt_cache["blocks"][layer]["resid_post"][:, QUERY_POSITION]

        patched_logits = model(clean_tokens, residual_patch=(layer, QUERY_POSITION, corrupt_query))
        targeted.append(
            (_action_margin(patched_logits, corrupt_action, clean_action) - clean_margin) / denominator
        )

        wrong_logits = model(clean_tokens, residual_patch=(layer, 2, corrupt_query))
        wrong_position.append(
            (_action_margin(wrong_logits, corrupt_action, clean_action) - clean_margin) / denominator
        )

        delta = corrupt_query - clean_query
        random_delta = t.randn(delta.shape, generator=generator, device=device)
        random_delta = F.normalize(random_delta, dim=-1) * delta.norm(dim=-1, keepdim=True)
        random_logits = model(
            clean_tokens,
            residual_patch=(layer, QUERY_POSITION, clean_query + random_delta),
        )
        matched_random.append(
            (_action_margin(random_logits, corrupt_action, clean_action) - clean_margin) / denominator
        )

        other_maze, _, other_corrupt_index = pairs[(pair_index + 1) % len(pairs)]
        other_tokens = _state_tokens(other_maze, other_maze.path[other_corrupt_index], device)
        _, other_cache = model(other_tokens, return_cache=True)
        other_query = other_cache["blocks"][layer]["resid_post"][:, QUERY_POSITION]
        unrelated_logits = model(
            clean_tokens,
            residual_patch=(layer, QUERY_POSITION, other_query),
        )
        unrelated.append(
            (_action_margin(unrelated_logits, corrupt_action, clean_action) - clean_margin)
            / denominator
        )

        # Does a donor with the same next action work without sharing this maze?
        donor = next((
            (candidate, second) for candidate, _, second in pairs
            if _maze_key(candidate) != _maze_key(maze)
            and action_between(candidate.path[second], candidate.path[second + 1]) == corrupt_action
        ), None)
        if donor is not None:
            donor_maze, donor_index = donor
            _, donor_cache = model(
                _state_tokens(donor_maze, donor_maze.path[donor_index], device), return_cache=True
            )
            donor_query = donor_cache["blocks"][layer]["resid_post"][:, QUERY_POSITION]
            donor_logits = model(clean_tokens, residual_patch=(layer, QUERY_POSITION, donor_query))
            same_action.append(
                (_action_margin(donor_logits, corrupt_action, clean_action) - clean_margin) / denominator
            )
            same_action_targeted.append(targeted[-1])

    first_maze, first_clean, first_corrupt = pairs[0]
    heatmap = residual_patching_heatmap(model, first_maze, first_clean, first_corrupt)
    if not same_action:
        raise RuntimeError("no same-action donors from different maze layouts are available")
    return PatchingResult(
        heatmap=heatmap,
        targeted_effect=float(t.tensor(targeted).median()),
        targeted_positive_fraction=float((t.tensor(targeted) > 0).float().mean()),
        wrong_position_effect=float(t.tensor(wrong_position).abs().median()),
        matched_norm_random_effect=float(t.tensor(matched_random).abs().median()),
        unrelated_route_effect=float(t.tensor(unrelated).median()),
        same_action_other_maze_effect=float(t.tensor(same_action).median()),
        same_action_targeted_effect=float(t.tensor(same_action_targeted).median()),
        same_action_pair_count=len(same_action),
        pair_count=len(pairs),
    )


def _rollout_suite(
    model: TinyMazeTransformer,
    mazes: tuple[Maze, ...],
    *,
    constrain_legal: bool,
) -> tuple[Rollout, ...]:
    return tuple(
        rollout_model(model, maze, constrain_legal=constrain_legal) for maze in mazes
    )


def run_signature_experiment(
    *,
    device: str | t.device = "cpu",
    seed: int = 0,
    train_maze_count: int = 420,
    heldout_maze_count: int = 100,
    epochs: int = 12,
    d_model: int = 64,
) -> ExperimentArtifacts:
    """Train, probe, intervene, and score paths on disjoint generated mazes."""

    device = t.device(device)
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats(device)
    train_mazes = generate_maze_collection(train_maze_count, seed=seed, require_detour_fraction=0.25)
    train_keys = {_maze_key(maze) for maze in train_mazes}
    heldout_mazes = generate_maze_collection(
        heldout_maze_count,
        seed=seed + 100_000,
        require_detour_fraction=0.5,
        excluded_keys=train_keys,
    )
    density_ood = generate_maze_collection(
        60,
        seed=seed + 200_000,
        obstacle_density=0.28,
        require_detour_fraction=0.5,
        excluded_keys=train_keys,
    )
    size_ood = generate_maze_collection(
        40,
        seed=seed + 300_000,
        size=6,
        obstacle_density=0.16,
        require_detour_fraction=0.25,
    )
    train_data = build_state_dataset(train_mazes)
    heldout_data = build_state_dataset(heldout_mazes)
    model, training = train_maze_transformer(
        train_data,
        heldout_data,
        device=device,
        seed=seed,
        epochs=epochs,
        d_model=d_model,
    )
    probes = layerwise_probe_metrics(model, train_data, heldout_data, seed=seed + 1)
    patching = route_state_intervention_metrics(model, heldout_mazes, seed=seed + 2)
    evaluation_mazes = heldout_mazes[:60]
    model_rollouts = _rollout_suite(model, evaluation_mazes, constrain_legal=True)
    raw_model_rollouts = _rollout_suite(model, evaluation_mazes, constrain_legal=False)
    template_rollouts = tuple(template_rollout(maze) for maze in evaluation_mazes)
    density_rollouts = _rollout_suite(
        model, density_ood[:40], constrain_legal=True
    )
    size_rollouts = _rollout_suite(model, size_ood[:30], constrain_legal=True)
    model_scores = score_rollouts(model_rollouts, evaluation_mazes)
    raw_model_scores = score_rollouts(raw_model_rollouts, evaluation_mazes)
    template_scores = score_rollouts(template_rollouts, evaluation_mazes)
    density_scores = score_rollouts(density_rollouts, density_ood[:40])
    size_scores = score_rollouts(size_rollouts, size_ood[:30])
    shuffled_accuracy = shuffled_maze_accuracy(model, heldout_data, seed=seed + 3)
    detour_accuracy = detour_action_accuracy(model, heldout_data)
    shuffled_detour_accuracy = detour_action_accuracy(
        model, heldout_data, shuffled_mazes=True, seed=seed + 3
    )
    final_action_accuracy = action_accuracy(model, heldout_data)
    template_overlap = len(train_keys & {_maze_key(maze) for maze in heldout_mazes})
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = t.cuda.max_memory_allocated(device) / 2**30

    metrics: dict[str, float | int | bool] = {
        "heldout_action_accuracy": final_action_accuracy,
        "shuffled_maze_action_accuracy": shuffled_accuracy,
        "detour_action_accuracy": detour_accuracy,
        "shuffled_maze_detour_accuracy": shuffled_detour_accuracy,
        "detour_shuffled_maze_margin": detour_accuracy - shuffled_detour_accuracy,
        "legal_path_rate": model_scores.legal_path_rate,
        "goal_reach_rate": model_scores.goal_reach_rate,
        "optimal_path_rate": model_scores.optimal_path_rate,
        "mean_excess_steps": model_scores.mean_excess_steps,
        "illegal_move_rate": model_scores.illegal_move_rate,
        "raw_goal_reach_rate": raw_model_scores.goal_reach_rate,
        "raw_optimal_path_rate": raw_model_scores.optimal_path_rate,
        "raw_illegal_move_rate": raw_model_scores.illegal_move_rate,
        "template_goal_reach_rate": template_scores.goal_reach_rate,
        "template_optimal_path_rate": template_scores.optimal_path_rate,
        "template_illegal_move_rate": template_scores.illegal_move_rate,
        "goal_reach_margin_over_template": (
            model_scores.goal_reach_rate - template_scores.goal_reach_rate
        ),
        "density_ood_goal_reach_rate": density_scores.goal_reach_rate,
        "density_ood_optimal_path_rate": density_scores.optimal_path_rate,
        "size_ood_goal_reach_rate": size_scores.goal_reach_rate,
        "size_ood_optimal_path_rate": size_scores.optimal_path_rate,
        "best_distance_probe_accuracy": max(probes.distance_accuracy),
        "best_legal_degree_probe_accuracy": max(probes.legal_degree_accuracy),
        "best_action_probe_accuracy": max(probes.action_accuracy),
        "random_label_distance_probe_accuracy": max(probes.random_label_distance_accuracy),
        "manhattan_distance_accuracy": probes.manhattan_distance_accuracy,
        "distance_probe_random_label_margin": (
            max(probes.distance_accuracy) - max(probes.random_label_distance_accuracy)
        ),
        "targeted_patch_effect": patching.targeted_effect,
        "targeted_patch_positive_fraction": patching.targeted_positive_fraction,
        "wrong_position_patch_effect": patching.wrong_position_effect,
        "matched_norm_random_patch_effect": patching.matched_norm_random_effect,
        "unrelated_route_patch_effect": patching.unrelated_route_effect,
        "same_action_other_maze_patch_effect": patching.same_action_other_maze_effect,
        "same_action_targeted_patch_effect": patching.same_action_targeted_effect,
        "same_action_patch_pair_count": patching.same_action_pair_count,
        "targeted_patch_wrong_position_margin": (
            patching.targeted_effect - patching.wrong_position_effect
        ),
        "targeted_patch_matched_random_margin": (
            patching.targeted_effect - patching.matched_norm_random_effect
        ),
        "patch_pair_count": patching.pair_count,
        "train_heldout_template_overlap": template_overlap,
        "peak_vram_gb": peak_vram_gb,
    }
    return ExperimentArtifacts(
        model=model,
        train_mazes=train_mazes,
        heldout_mazes=heldout_mazes,
        density_ood_mazes=density_ood,
        size_ood_mazes=size_ood,
        training=training,
        probes=probes,
        patching=patching,
        model_rollouts=model_rollouts,
        bfs_paths=tuple(maze.path for maze in evaluation_mazes),
        metrics=metrics,
    )


def validate_experiment(
    artifacts: ExperimentArtifacts,
    *,
    require_cuda: bool = False,
    max_vram_gb: float = 24.0,
) -> None:
    metrics = artifacts.metrics
    if require_cuda:
        assert next(artifacts.model.parameters()).is_cuda, (
            "GPU verification requires CUDA; CPU fallback is forbidden."
        )
    checks = {
        "zero train/heldout overlap": metrics["train_heldout_template_overlap"] == 0,
        "heldout action >= 0.72": metrics["heldout_action_accuracy"] >= 0.72,
        "true-detour shuffled-wall margin >= 0.04": metrics["detour_action_accuracy"] >= metrics["shuffled_maze_detour_accuracy"] + 0.04,
        "legal paths >= 0.99": metrics["legal_path_rate"] >= 0.99,
        "goal reach >= 0.58": metrics["goal_reach_rate"] >= 0.58,
        "optimal paths >= 0.48": metrics["optimal_path_rate"] >= 0.48,
        "goal reach template margin >= 0.08": metrics["goal_reach_rate"] >= metrics["template_goal_reach_rate"] + 0.08,
        "distance probe >= 0.62": metrics["best_distance_probe_accuracy"] >= 0.62,
        "distance probe random-label margin >= 0.25": metrics["best_distance_probe_accuracy"] >= metrics["random_label_distance_probe_accuracy"] + 0.25,
        "targeted patch >= 0.15": metrics["targeted_patch_effect"] >= 0.15,
        "positive patch pairs >= 0.85": metrics["targeted_patch_positive_fraction"] >= 0.85,
        "patch wrong-position margin >= 0.10": metrics["targeted_patch_effect"] >= metrics["wrong_position_patch_effect"] + 0.10,
        "patch random-direction margin >= 0.10": metrics["targeted_patch_effect"] >= metrics["matched_norm_random_patch_effect"] + 0.10,
        "VRAM within declared budget": metrics["peak_vram_gb"] <= max_vram_gb,
    }
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, "Failed numerical checks: " + "; ".join(failed)


def _serializable_report(artifacts: ExperimentArtifacts) -> dict[str, Any]:
    return {
        **artifacts.metrics,
        "training": {
            "train_loss": artifacts.training.train_loss,
            "heldout_action_accuracy": artifacts.training.heldout_action_accuracy,
            "checkpoint_epochs": artifacts.training.checkpoint_epochs,
        },
        "probes": asdict(artifacts.probes),
        "patching": {
            **asdict(artifacts.patching),
            "heatmap": artifacts.patching.heatmap.tolist(),
        },
        "tests_passed": False,  # A model experiment does not execute the test suite.
        "contract_passed": True,
        "accepted": False,  # Release review and course wiring remain pending.
    }


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    if not cpu:
        return run_gpu_test()
    artifacts = run_signature_experiment(
        device="cpu",
        train_maze_count=260,
        heldout_maze_count=70,
        epochs=10,
        d_model=56,
    )
    validate_experiment(artifacts)
    report = _serializable_report(artifacts)
    report.update({"cuda_available": False, "within_vram_budget": True})
    return report


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    if not t.cuda.is_available():
        raise RuntimeError("14.4 GPU verification requires CUDA; CPU fallback is forbidden")
    artifacts = run_signature_experiment(
        device="cuda",
        train_maze_count=900,
        heldout_maze_count=180,
        epochs=18,
        d_model=72,
    )
    validate_experiment(artifacts, require_cuda=True, max_vram_gb=max_vram_gb)
    report = _serializable_report(artifacts)
    report.update(
        {
            "cuda_available": True,
            "device": t.cuda.get_device_name(),
            "gpu_name": t.cuda.get_device_name(),
            "torch_version": t.__version__,
            "cuda_version": t.version.cuda,
            "within_vram_budget": artifacts.metrics["peak_vram_gb"] <= max_vram_gb,
        }
    )
    return report


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    return run_gpu_test(max_vram_gb=max_vram_gb)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2))
