# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = './data/'
C.DATASET.NAME = 'ILSVRC2012'
C.DATASET.NUM_CLASSES = 1000
C.DATASET.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.BASE_LR = 0.05
C.SOLVER.LR_MILESTONES = [30, 60, 90]
C.SOLVER.LR_GAMMA = 0.1
C.SOLVER.WEIGHT_DECAY = 1e-4
C.SOLVER.MOMENTUM = 0.9
C.SOLVER.DAMPENING = False
C.SOLVER.NESTEROV = False

C.SOLVER.TRAIN_BATCH_SIZE = 128
C.SOLVER.TEST_BATCH_SIZE = 200

C.SOLVER.MAX_EPOCHS = 100

C.SOLVER.WARMUP = False
C.SOLVER.WARMUP_EPOCH = 5
C.SOLVER.WARMUP_FACTOR = 0.2

C.SOLVER.DISAG_COEFF = 0.1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.OUTPUT_DIR = './outputs/cls'
