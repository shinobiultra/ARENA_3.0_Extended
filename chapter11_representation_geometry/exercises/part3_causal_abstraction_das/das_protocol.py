"""Frozen native final protocol, selected before any final-model evaluation.

Selection sources: native-screen-v1 (six sites, 120 updates), then
native-ranks-v1 (ranks 1, 2, 4, 8, 240 updates), both train/development only.
Unconditional cross-name changed-role IIA selected block 9 input, end position,
rank 8. No final outcomes entered these choices.
"""

LAYER = 9
POSITION = "end"
RANK = 8
STEPS = 240
BATCH_SIZE = 16
LEARNING_RATE = 0.01
SEEDS = (0, 1, 2)
SHUFFLE_SEEDS = (400, 401, 402)
RANDOM_SEEDS = tuple(range(700, 732))
PRIMARY_SUBGROUP = "cross_changed"
PRIMARY_METRIC = "full_vocabulary_interchange_accuracy"
RIDGE = 0.1

# Fixed examples are the first cross-name, changed-role pair in each final
# template. Choose before model access; never choose favorable final outcomes.
DISPLAY_TEMPLATE_IDS = (0, 1, 2)
