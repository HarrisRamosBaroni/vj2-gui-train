# ==============================================================================
# VJ2-GUI UNIFIED ACTION PIPELINE - CANONICAL CONFIGURATION
# ==============================================================================
# This file serves as the single source of truth for all pipeline parameters
# as defined in PIPELINE_DESIGN.md. Modifying these values will propagate
# the changes throughout the entire agent system.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Canonical Frequencies (in Hertz)
# ------------------------------------------------------------------------------
# The rates at which different components of the system should run.

# F_action: The target execution frequency of individual actions on the device.
# This is the "metronome" of the system, fixed at the hardware's ideal rate.
ACTION_HZ = 500.0

# F_observe: The rate at which the agent captures and processes screen observations.
# This is the primary system bottleneck, limited by screen capture and model inference.
OBSERVER_HZ = 4.0

# F_plan: The rate at which the planner generates new action sequences.
# Typically F_plan = F_observe / 2, as the planner requires two observations
# (a start and a target) to form a planning problem.
PLANNER_HZ = OBSERVER_HZ / 2.0

# F_execute: The polling rate for the high-precision executor loop.
# Must be significantly higher than ACTION_HZ (nyquist freq) to ensure accurate timing. At least twice nyquist?
EXECUTOR_POLLING_HZ = 1000.0


# ------------------------------------------------------------------------------
# 2. Canonical Horizons and Dimensions
# ------------------------------------------------------------------------------
# Defines the shape and size of the action plans.

# The number of "Action Batches" the planner optimizes over. This is the `horizon`
# in the MPC problem. An "Action Batch" is the unit of optimization.
PLANNING_HORIZON_BATCHES = 1

# The number of atomic actions contained within a single "Action Batch".
# The total number of actions generated per plan is:
# N_actions = PLANNING_HORIZON_BATCHES * ACTIONS_PER_BATCH
# To meet the device's 240Hz sampling rate, we need: N_actions * PLANNER_HZ >= ACTION_HZ
# Represents the number of atomic actions that correspond to a single state
# transition. For example, if the planner runs at 2Hz (one state update every
# 0.5s) and the action frequency is 500Hz, then one "Action Batch" must contain
# 0.5s * 500 actions/s = 250 atomic actions.
ACTIONS_PER_BATCH = int(ACTION_HZ / PLANNER_HZ)

# The dimensionality of a single Normalized Atomic Action (NAA).
# Format: [x_norm, y_norm, touch_state_norm]
ACTION_DIM = 3

# Action autoencoder latent action dimension
LATENT_ACTION_DIM = 20

#
# The number of observations (frames) in a single training window (4s).
# MAX_OBSERVATION_CONTEXT = 16
# WINDOW_TIME = int(MAX_OBSERVATION_CONTEXT / OBSERVER_HZ)  # 16/4hz=4s
# OBSERVATIONS_PER_WINDOW = WINDOW_TIME * OBSERVER_HZ # 4*4 = 16 so full context, duh
OBSERVATIONS_PER_WINDOW = int(4.0 * OBSERVER_HZ)

# Number of latent states per window (half OBSERVATIONS_PER_WINDOW since tubelets are size 2)
LATENT_STATES_PER_WINDOW = OBSERVATIONS_PER_WINDOW // 2   # 8

# The number of action blocks corresponding to a training window.
# One action block is the transition between two latent states, and since there
# are 8 latent states in a 4s window, there are 7 transitions.
ACTION_BLOCKS_PER_WINDOW = (OBSERVATIONS_PER_WINDOW // 2) - 1

# The number of autoregressive steps to perform for the rollout loss.
ROLLOUT_HORIZON = 2

# ------------------------------------------------------------------------------
# 3. System and Device Configuration
# ------------------------------------------------------------------------------

# The absolute path to the compiled C event injector on the Android device.
NATIVE_INJECTOR_PATH = "/data/local/tmp/event_injector"

# The minimum number of frames the VJEPA encoder requires to create a context. Max is 16.
ENCODER_CLIP_LEN = 2

# The minimum number of frames the VJEPA encoder requires to create a context. Max is 16.
NUM_CONTEXT_FRAMES = 16

# The number of gradient descent steps the planner takes to find a good plan.
PLANNING_ITERATIONS = 10

# Toggle for adjusting planner timestamps to account for planning time.
# Set to True to make action dispatch more consistent with ACTION_HZ
# when PLANNING_ITERATIONS > 0.
ADJUST_PLANNER_TIMESTAMPS = False
# ADJUST_PLANNER_TIMESTAMPS = True
