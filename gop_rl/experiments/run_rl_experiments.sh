#!/usr/bin/env bash
set -e  # exit on any error

# ——— Common settings ———
INPUT_TYPE="state"
N_CYCLES=1
TRAIN_STEPS=10000
EVAL_FREQ=1000
NUM_ENVS=100
DEVICE="cuda"

# ——— Pendulum-specific ———
ENV_PEND="Pendulum-v1"
MAX_EP_LEN_PEND=200
WARM_UP_PEND=0
DATA_DIR_PEND="data/pendulum"
OUTPUT_DIR_PEND="outputs/pendulum"

# ——— Ant-specific ———
ENV_ANT="Ant-v4"
MAX_EP_LEN_ANT=1000
WARM_UP_ANT=40
DATA_DIR_ANT="data/ant"
OUTPUT_DIR_ANT="outputs/ant"

# ——— Sweep over these ———
SCHEMES=(bias "warm-start" dcc)
AGENTS=(gaussian mle cnf)   # add 'mle' here if you want to sweep it too

for AGENT in "${AGENTS[@]}"; do

  if [[ "$AGENT" == "gaussian" ]]; then
    # one run per env, no insertion‐scheme loop
    echo "============================================"
    echo "Agent: $AGENT | Env: Pendulum"
    echo "============================================"
    python3 -m gop_rl.experiments.run_rl \
      --env_name           "$ENV_PEND" \
      --exploration_type   "$AGENT" \
      --input_type         "$INPUT_TYPE" \
      --n_cycles           "$N_CYCLES" \
      --training_steps     "$TRAIN_STEPS" \
      --max_episode_length "$MAX_EP_LEN_PEND" \
      --warm_up            "$WARM_UP_PEND" \
      --n_grad_steps       1 \
      --batch_size         1024 \
      --num_envs           "$NUM_ENVS" \
      --eval_freq          "$EVAL_FREQ" \
      --device             "$DEVICE" \
      --data_dir           "$DATA_DIR_PEND" \
      --output_dir         "$OUTPUT_DIR_PEND/$AGENT"
    echo "→ Pendulum [$AGENT] done."

    echo "============================================"
    echo "Agent: $AGENT | Env: Ant"
    echo "============================================"
    python3 -m gop_rl.experiments.run_rl \
      --env_name           "$ENV_ANT" \
      --exploration_type   "$AGENT" \
      --input_type         "$INPUT_TYPE" \
      --n_cycles           "$N_CYCLES" \
      --training_steps     "$TRAIN_STEPS" \
      --max_episode_length "$MAX_EP_LEN_ANT" \
      --warm_up            "$WARM_UP_ANT" \
      --n_grad_steps       1 \
      --batch_size         1024 \
      --num_envs           "$NUM_ENVS" \
      --eval_freq          "$EVAL_FREQ" \
      --device             "$DEVICE" \
      --data_dir           "$DATA_DIR_ANT" \
      --output_dir         "$OUTPUT_DIR_ANT/$AGENT"
    echo "→ Ant      [$AGENT] done."

  else
    # for cnf (and mle if added), sweep insertion schemes
    for SCHEME in "${SCHEMES[@]}"; do

      # echo "============================================"
      # echo "Agent: $AGENT | Scheme: $SCHEME | Env: Pendulum"
      # echo "============================================"
      # python3 -m gop_rl.experiments.run_rl \
      #   --env_name           "$ENV_PEND" \
      #   --exploration_type   "$AGENT" \
      #   --input_type         "$INPUT_TYPE" \
      #   --insertion_scheme   "$SCHEME" \
      #   --n_cycles           "$N_CYCLES" \
      #   --training_steps     "$TRAIN_STEPS" \
      #   --max_episode_length "$MAX_EP_LEN_PEND" \
      #   --warm_up            "$WARM_UP_PEND" \
      #   --n_grad_steps       3 \
      #   --batch_size         1024 \
      #   --num_envs           "$NUM_ENVS" \
      #   --eval_freq          "$EVAL_FREQ" \
      #   --device             "$DEVICE" \
      #   --data_dir           "$DATA_DIR_PEND" \
      #   --output_dir         "$OUTPUT_DIR_PEND/$AGENT/$SCHEME"
      # echo "→ Pendulum [$AGENT / $SCHEME] done."

      echo "============================================"
      echo "Agent: $AGENT | Scheme: $SCHEME | Env: Ant"
      echo "============================================"
      python3 -m gop_rl.experiments.run_rl \
        --env_name           "$ENV_ANT" \
        --exploration_type   "$AGENT" \
        --input_type         "$INPUT_TYPE" \
        --insertion_scheme   "$SCHEME" \
        --n_cycles           "$N_CYCLES" \
        --training_steps     "$TRAIN_STEPS" \
        --max_episode_length "$MAX_EP_LEN_ANT" \
        --warm_up            "$WARM_UP_ANT" \
        --n_grad_steps       3 \
        --batch_size         1024 \
        --num_envs           "$NUM_ENVS" \
        --eval_freq          "$EVAL_FREQ" \
        --device             "$DEVICE" \
        --data_dir           "$DATA_DIR_ANT" \
        --output_dir         "$OUTPUT_DIR_ANT/$AGENT/$SCHEME"
      echo "→ Ant      [$AGENT / $SCHEME] done."

    done
  fi

done

echo "All experiments finished successfully."
