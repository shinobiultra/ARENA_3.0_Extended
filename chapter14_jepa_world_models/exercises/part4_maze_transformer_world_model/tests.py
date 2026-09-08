"""Visible semantic tests for [14.4] Maze Transformers and Latent Planning."""

from __future__ import annotations

import torch as t
import pytest
from torch import nn
from dataclasses import replace
import ast
import json
import re
import hashlib
from pathlib import Path

from . import solutions


def _passed(name: str) -> None:
    print(f"All tests in `{name}` passed!")


def _hand_maze() -> solutions.Maze:
    walls = t.tensor(
        [
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=t.bool,
    )
    path = solutions.bfs_shortest_path(walls, (0, 0), (4, 4))
    assert path is not None
    return solutions.Maze(walls, (0, 0), (4, 4), path)


def test_bfs_shortest_path_hand_oracle(fn=solutions.bfs_shortest_path) -> None:
    maze = _hand_maze()
    expected = (
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    )
    assert fn(maze.walls, maze.start, maze.goal) == expected
    assert len(expected) - 1 == 8
    _passed("test_bfs_shortest_path_hand_oracle")


def test_distance_map_exact(fn=solutions.distance_map) -> None:
    maze = _hand_maze()
    distances = fn(maze.walls, maze.goal)
    assert distances[maze.goal] == 0
    assert distances[maze.start] == maze.distance
    assert distances[0, 2] == -1
    for first, second in zip(maze.path[:-1], maze.path[1:]):
        assert distances[first] == distances[second] + 1
    _passed("test_distance_map_exact")


def test_generate_maze_deterministic(fn=solutions.generate_maze) -> None:
    first = fn(19, require_detour=True)
    second = fn(19, require_detour=True)
    assert t.equal(first.walls, second.walls)
    assert first.start == second.start and first.goal == second.goal and first.path == second.path
    manhattan = abs(first.start[0] - first.goal[0]) + abs(first.start[1] - first.goal[1])
    assert first.distance > manhattan
    _passed("test_generate_maze_deterministic")


def test_action_between_and_legality(fn=solutions.action_between) -> None:
    assert [fn((2, 2), cell) for cell in ((1, 2), (2, 3), (3, 2), (2, 1))] == [0, 1, 2, 3]
    maze = _hand_maze()
    assert solutions.apply_action(maze.walls, (0, 0), 1) == (0, 1)
    assert solutions.apply_action(maze.walls, (0, 0), 2) is None
    _passed("test_action_between_and_legality")


def test_tokenize_maze_state(fn=solutions.tokenize_maze_state) -> None:
    maze = _hand_maze()
    tokens = fn(maze, maze.start)
    assert tokens.shape == (solutions.SEQUENCE_LENGTH,)
    assert tokens[0] == solutions.BOS and tokens[1] == solutions.SIZE_5
    grid = tokens[2:-1].view(7, 7)
    assert grid[maze.start] == solutions.CURRENT
    assert grid[maze.goal] == solutions.GOAL
    assert grid[0, 2] == solutions.WALL
    assert (grid[5:] == solutions.WALL).all() and (grid[:, 5:] == solutions.WALL).all()
    assert tokens[-1] == solutions.QUERY
    _passed("test_tokenize_maze_state")


def test_build_state_dataset_bfs_policy(fn=solutions.build_state_dataset) -> None:
    maze = _hand_maze()
    dataset = fn((maze,))
    distances = solutions.distance_map(maze.walls, maze.goal)
    reachable_non_goal = int(((distances > 0) & ~maze.walls).sum())
    assert len(dataset.tokens) == reachable_non_goal
    for state in dataset.states:
        next_cell = solutions.apply_action(maze.walls, state.current, state.action)
        assert next_cell is not None
        assert distances[next_cell] == distances[state.current] - 1
        assert state.legal_degree == len(solutions.legal_neighbors(maze.walls, state.current))
    _passed("test_build_state_dataset_bfs_policy")


def test_detour_labels_exclude_equally_short_ties(fn=solutions.build_state_dataset) -> None:
    walls = t.zeros(5, 5, dtype=t.bool)
    maze = solutions.Maze(walls, (2, 0), (0, 4), solutions.bfs_shortest_path(walls, (2, 0), (0, 4)))
    dataset = fn((maze,))
    state = next(state for state in dataset.states if state.current == (2, 0))
    assert state.action == 0  # Canonical BFS prefers up over right.
    assert solutions.template_action(state.current, state.goal) == 1
    assert state.template_is_shortest, "Right is equally short in the open maze; this is not a detour."
    blocked = walls.clone()
    blocked[2, 1] = True
    blocked_maze = replace(maze, walls=blocked, path=solutions.bfs_shortest_path(blocked, maze.start, maze.goal))
    blocked_state = next(state for state in fn((blocked_maze,)).states if state.current == (2, 0))
    assert not blocked_state.template_is_shortest, "The wall-blind right move now hits a wall."
    _passed("test_detour_labels_exclude_equally_short_ties")


def test_attention_is_causal(attention_cls=solutions.CausalSelfAttention) -> None:
    t.manual_seed(0)
    attention = attention_cls(d_model=12, n_heads=3)
    first = t.randn(2, 6, 12)
    second = first.clone()
    second[:, 4:] = t.randn_like(second[:, 4:]) * 10
    first_output, first_pattern, _ = attention(first)
    second_output, _, _ = attention(second)
    t.testing.assert_close(first_output[:, :4], second_output[:, :4], rtol=1e-5, atol=1e-6)
    assert first_pattern[..., t.triu(t.ones(6, 6, dtype=t.bool), diagonal=1)].abs().max() == 0
    with t.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        attention.qkv.weight[24:] = t.eye(12)
        attention.output.weight.copy_(t.eye(12))
        attention.output.bias.zero_()
    actual, _, _ = attention(first)
    expected = first.cumsum(dim=1) / t.arange(1, 7)[None, :, None]
    t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _passed("test_attention_is_causal")


def test_model_cache_and_residual_patch(model_cls=solutions.TinyMazeTransformer) -> None:
    t.manual_seed(0)
    model = model_cls(d_model=24, n_layers=2, n_heads=4, d_mlp=48)
    first = solutions.tokenize_maze_state(_hand_maze(), (0, 0)).unsqueeze(0)
    second = solutions.tokenize_maze_state(_hand_maze(), (2, 2)).unsqueeze(0)
    first_logits, first_cache = model(first, return_cache=True)
    second_logits, second_cache = model(second, return_cache=True)
    assert first_logits.shape == (1, 4)
    assert len(first_cache["blocks"]) == 2
    for block in first_cache["blocks"]:
        assert block["resid_post"].shape == (1, solutions.SEQUENCE_LENGTH, 24)
        assert block["pattern"].shape == (1, 4, solutions.SEQUENCE_LENGTH, solutions.SEQUENCE_LENGTH)
    replacement = second_cache["blocks"][-1]["resid_post"][:, solutions.QUERY_POSITION]
    patched = model(first, residual_patch=(1, solutions.QUERY_POSITION, replacement))
    t.testing.assert_close(patched, second_logits, rtol=1e-5, atol=1e-6)
    _passed("test_model_cache_and_residual_patch")


def test_training_and_checkpointing(train_fn=solutions.train_maze_transformer) -> None:
    train_mazes = solutions.generate_maze_collection(24, seed=30)
    heldout_mazes = solutions.generate_maze_collection(
        8,
        seed=10_000,
        excluded_keys={solutions._maze_key(maze) for maze in train_mazes},
    )
    train = solutions.build_state_dataset(train_mazes)
    heldout = solutions.build_state_dataset(heldout_mazes)
    t.manual_seed(0)
    initial = solutions.TinyMazeTransformer(d_model=24, n_layers=2)
    with t.no_grad():
        initial_loss = float(t.nn.functional.cross_entropy(initial(train.tokens), train.actions))
    model, result = train_fn(
        train,
        heldout,
        epochs=3,
        d_model=24,
        n_layers=2,
        batch_size=64,
    )
    assert len(result.train_loss) == len(result.heldout_action_accuracy) == 3
    assert result.checkpoint_epochs == (1, 2, 3)
    assert all(t.isfinite(t.tensor(result.train_loss)))
    first_name = next(iter(result.checkpoints[0]))
    assert result.checkpoints[0][first_name].device.type == "cpu"
    assert result.checkpoints[0][first_name].data_ptr() != model.state_dict()[first_name].data_ptr()
    assert not t.equal(result.checkpoints[0][first_name], result.checkpoints[-1][first_name])
    for name, value in model.state_dict().items():
        t.testing.assert_close(value.cpu(), result.checkpoints[-1][name])
    with t.no_grad():
        final_loss = float(t.nn.functional.cross_entropy(model(train.tokens), train.actions))
    assert final_loss < initial_loss - 0.01, "Actual trained predictions must improve, not just the reported history."
    _passed("test_training_and_checkpointing")


def test_template_rollout_scores_validity_separately(
    template_fn=solutions.template_rollout, score_fn=solutions.score_rollouts,
) -> None:
    open_walls = t.zeros(5, 5, dtype=t.bool)
    path = solutions.bfs_shortest_path(open_walls, (0, 0), (0, 4))
    assert path is not None
    open_maze = solutions.Maze(open_walls, (0, 0), (0, 4), path)
    good = template_fn(open_maze)
    assert good.legal and good.reached_goal and good.optimal

    blocked_walls = open_walls.clone()
    blocked_walls[0, 1] = True
    detour = solutions.bfs_shortest_path(blocked_walls, (0, 0), (0, 4))
    assert detour is not None
    blocked_maze = solutions.Maze(blocked_walls, (0, 0), (0, 4), detour)
    bad = template_fn(blocked_maze)
    assert not bad.legal and not bad.reached_goal and not bad.optimal
    metrics = score_fn((good, bad), (open_maze, blocked_maze))
    assert metrics.legal_path_rate == metrics.goal_reach_rate == metrics.optimal_path_rate == 0.5
    _passed("test_template_rollout_scores_validity_separately")


def test_rollout_scores_replay_actions(fn=solutions.score_rollouts) -> None:
    walls = t.zeros(5, 5, dtype=t.bool)
    maze = solutions.Maze(walls, (0, 0), (0, 2), ((0, 0), (0, 1), (0, 2)))
    longer = solutions.Rollout(((0, 0), (1, 0), (1, 1), (1, 2), (0, 2)), (2, 1, 1, 0), True, True, True)
    scored = fn((longer,), (maze,))
    assert scored.legal_path_rate == scored.goal_reach_rate == 1.0
    assert scored.optimal_path_rate == 0.0 and scored.mean_excess_steps == 2.0
    fabricated_success = solutions.Rollout(((0, 0),), (), True, True, True)
    assert fn((fabricated_success,), (maze,)).goal_reach_rate == 0.0
    illegal = solutions.Rollout(((0, 0),), (0,), True, True, True)
    scored = fn((illegal,), (maze,))
    assert scored.illegal_move_rate == 1.0 and scored.goal_reach_rate == 0.0
    with pytest.raises(ValueError, match="cell trace"):
        fn((replace(longer, cells=((0, 0), (0, 2))),), (maze,))
    _passed("test_rollout_scores_replay_actions")


def test_raw_and_constrained_rollout_use_learner_policy(fn=solutions.rollout_model) -> None:
    class RightFirstPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(t.tensor(0.0))

        def forward(self, tokens):
            return t.tensor([[3., 4., 2., 1.]]).expand(len(tokens), -1) + self.anchor * 0

    walls = t.zeros(5, 5, dtype=t.bool)
    walls[0, 1] = True
    maze = solutions.Maze(walls, (0, 0), (0, 4), solutions.bfs_shortest_path(walls, (0, 0), (0, 4)))
    raw = fn(RightFirstPolicy(), maze, constrain_legal=False)
    constrained = fn(RightFirstPolicy(), maze, constrain_legal=True)
    assert raw.actions == (1,) and not raw.legal and not raw.reached_goal
    assert constrained.cells == ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (0, 4))
    assert constrained.legal and constrained.reached_goal and constrained.optimal
    assert fn(RightFirstPolicy(), maze, max_steps=0).actions == ()
    _passed("test_raw_and_constrained_rollout_use_learner_policy")


def test_ridge_probe_toy_oracle(fn=solutions.ridge_probe_accuracy) -> None:
    labels = t.arange(3).repeat_interleave(20)
    features = t.nn.functional.one_hot(labels, 3).float()
    train_indices = t.tensor([index for index in range(60) if index % 4 != 0])
    test_indices = t.tensor([index for index in range(60) if index % 4 == 0])
    accuracy = fn(
        features[train_indices],
        labels[train_indices],
        features[test_indices],
        labels[test_indices],
        n_classes=3,
    )
    assert accuracy == 1.0
    _passed("test_ridge_probe_toy_oracle")


def test_residual_patching_heatmap_has_exact_final_recovery(fn=solutions.residual_patching_heatmap) -> None:
    t.manual_seed(2)
    model = solutions.TinyMazeTransformer(d_model=24, n_layers=2, n_heads=4, d_mlp=48)
    maze = solutions.generate_maze(50, require_detour=True)
    actions = [solutions.action_between(a, b) for a, b in zip(maze.path[:-1], maze.path[1:])]
    pair = next(
        (first, second)
        for first in range(len(actions))
        for second in range(first + 1, len(actions))
        if actions[first] != actions[second]
    )
    heatmap = fn(model, maze, *pair)
    assert heatmap.shape == (2, solutions.SEQUENCE_LENGTH)
    t.testing.assert_close(heatmap[-1, -1], t.tensor(1.0), rtol=1e-4, atol=1e-4)
    _passed("test_residual_patching_heatmap_has_exact_final_recovery")


def test_shuffle_preserves_markers_and_input(fn=solutions.shuffled_maze_accuracy) -> None:
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(t.tensor(0.0))
            self.seen = []

        def forward(self, tokens):
            self.seen.append(tokens.detach().clone())
            return t.zeros(len(tokens), 4) + self.anchor * 0

    dataset = solutions.build_state_dataset(solutions.generate_maze_collection(8, seed=9))
    before = dataset.tokens.clone()
    model = Recorder()
    fn(model, dataset, seed=2)
    after = t.cat(model.seen)
    for marker in (solutions.CURRENT, solutions.GOAL, solutions.CURRENT_GOAL):
        assert t.equal(after == marker, before == marker), "Shuffling walls must preserve original markers exactly."
    assert (after[:, 2:-1] != before[:, 2:-1]).any(), "The wall control must actually change some layouts."
    t.testing.assert_close(dataset.tokens, before)
    _passed("test_shuffle_preserves_markers_and_input")


def test_same_action_donor_exposes_action_only_state(fn=solutions.route_state_intervention_metrics) -> None:
    class ActionOnlyState(nn.Module):
        """An exact policy oracle with only an action one-hot at the readout."""
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(t.tensor(0.0))
            self.n_layers = 2

        def forward(self, tokens, *, return_cache=False, residual_patch=None):
            encoded = []
            for row in tokens:
                grid = row[2:-1].reshape(7, 7)[:5, :5]
                start = tuple((grid == solutions.CURRENT).nonzero()[0].tolist())
                goal = tuple((grid == solutions.GOAL).nonzero()[0].tolist())
                path = solutions.bfs_shortest_path(grid == solutions.WALL, start, goal)
                action = solutions.action_between(path[0], path[1])
                encoded.append(t.nn.functional.one_hot(t.tensor(action), 4).float() * 4)
            x = t.zeros(len(tokens), solutions.SEQUENCE_LENGTH, 4)
            x[:, solutions.QUERY_POSITION] = t.stack(encoded)
            cache = {'blocks': []}
            for layer in range(self.n_layers):
                if residual_patch is not None and residual_patch[0] == layer:
                    _, position, replacement = residual_patch
                    x = x.clone()
                    x[:, position] = replacement
                cache['blocks'].append({'resid_post': x.clone()})
            logits = x[:, solutions.QUERY_POSITION] + self.anchor * 0
            return (logits, cache) if return_cache else logits

    mazes = solutions.generate_maze_collection(24, seed=50, require_detour_fraction=0.5)
    result = fn(ActionOnlyState(), mazes, seed=7)
    assert result.same_action_pair_count >= 4
    assert result.same_action_targeted_effect == pytest.approx(1.0)
    assert result.same_action_other_maze_effect == pytest.approx(1.0)
    assert result.wrong_position_effect == pytest.approx(0.0)
    assert result.targeted_effect == pytest.approx(1.0)
    _passed("test_same_action_donor_exposes_action_only_state")


def test_dropdown_answers_complete_the_paired_learner() -> None:
    section = Path(__file__).resolve().parent
    learner = json.loads(next(section.glob('*exercises.ipynb')).read_text())
    solved = json.loads(next(section.glob('*solutions.ipynb')).read_text())
    answer = ''
    completed = 0
    for cell, reference_cell in zip(learner['cells'], solved['cells'], strict=True):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if '<summary>Solution</summary>' in source:
                answer = re.search(r'<summary>Solution</summary>\s*```python\n(.*?)\n```', source, re.S)[1]
                ast.parse(answer)
            continue
        lines = source.splitlines(keepends=True)
        replacements = []
        for node in ast.parse(source).body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or 'NotImplementedError' not in ast.unparse(node):
                continue
            match = next(item for item in ast.parse(answer).body if getattr(item, 'name', None) == node.name)
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            answer_start = min([match.lineno, *(item.lineno for item in match.decorator_list)]) - 1
            replacement = '\n'.join(answer.splitlines()[answer_start:match.end_lineno]) + '\n'
            replacements.append((start, node.end_lineno, replacement))
        for start, end, replacement in reversed(replacements):
            lines[start:end] = [replacement]
        completed += bool(replacements)
        assert ast.dump(ast.parse(''.join(lines))) == ast.dump(ast.parse(''.join(reference_cell['source'])))
    assert completed == 12
    _passed("test_dropdown_answers_complete_the_paired_learner")


def test_cpu_smoke_signature_result() -> None:
    report = solutions.run_smoke_test(cpu=True)
    assert report["goal_reach_rate"] > report["template_goal_reach_rate"] + 0.08
    assert report["best_distance_probe_accuracy"] > report["random_label_distance_probe_accuracy"] + 0.25
    assert report["targeted_patch_effect"] > report["matched_norm_random_patch_effect"] + 0.1
    _passed("test_cpu_smoke_signature_result")


def test_reference_evidence_is_current() -> None:
    section = Path(__file__).resolve().parent
    root = section.parents[2]
    evidence = json.loads((section / 'expected_outputs/reference_metrics.json').read_text())
    assert evidence['protocol_version'] == 'maze-14.4-policy-controls-v2'
    assert evidence['accepted'] is False, "This candidate has not received course release acceptance."
    assert evidence['tests_passed']
    assert evidence['learner']['executed_code_cells'] > 20
    assert evidence['learner']['error_outputs'] == 0
    for name, expected in evidence['input_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Stale evidence: {name}'
    assert evidence['cpu']['contract_passed']
    if evidence['cuda'] is not None:
        assert evidence['cuda']['device'] == 'cuda'
        assert 0 < evidence['cuda']['peak_vram_gb'] <= 24
        assert isinstance(evidence['cuda']['contract_passed'], bool)
        assert evidence['cuda']['contract_passed'] or evidence['cuda']['contract_failure']
    _passed("test_reference_evidence_is_current")
