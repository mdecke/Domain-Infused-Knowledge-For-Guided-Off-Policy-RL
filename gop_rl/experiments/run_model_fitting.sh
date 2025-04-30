set -e          # abort on first error
set -u          # error on undefined vars

ENV_NAME="Ant-v4"
DATA_DIR="data/ant"
BASE_OUT="outputs/ant"

MODEL_TYPES=(mle cnf)
INPUT_TYPES=('state' 'state_action' 'prev_state_action' 'state_prev_state')

for MODEL in "${MODEL_TYPES[@]}"; do
  for INPUT in "${INPUT_TYPES[@]}"; do
    OUT_DIR="${BASE_OUT}"
    mkdir -p "${OUT_DIR}"

    echo "▶ ${ENV_NAME} | model=${MODEL} | input=${INPUT}"
    python3 -m gop_rl.experiments.fit_model \
        --env_name     "${ENV_NAME}" \
        --model_type   "${MODEL}" \
        --input_type   "${INPUT}" \
        --data_dir     "${DATA_DIR}" \
        --output_dir   "${OUT_DIR}" \
        --epoch 100 \
        --batch_size 5096 \
        --nb_traj 128\
        --n_grad_steps 20\
        --early_stopping 5 \
        --n_flows 20 \
        --n_cycles 1\
        --seed 42
  done
done

echo "✓ All Ant experiments completed."




# echo "✓ Starting Pendulum experiments."

# ENV_NAME="Pendulum-v1"
# DATA_DIR="data/pendulum"
# BASE_OUT="outputs/pendulum"

# MODEL_TYPES=('mle' 'cnf')
# INPUT_TYPES=('state' 'state_action' 'prev_state_action' 'state_prev_state')

# for MODEL in "${MODEL_TYPES[@]}"; do
#   for INPUT in "${INPUT_TYPES[@]}"; do
#     OUT_DIR="${BASE_OUT}"
#     mkdir -p "${OUT_DIR}"

#     echo "▶ ${ENV_NAME} | model=${MODEL} | input=${INPUT}"
#     python3 -m gop_rl.experiments.fit_model \
#         --env_name     "${ENV_NAME}" \
#         --model_type   "${MODEL}" \
#         --input_type   "${INPUT}" \
#         --data_dir     "${DATA_DIR}" \
#         --output_dir   "${OUT_DIR}" \
#         --nb_traj 32\
#         --n_grad_steps 20\
#         --n_cycles 1\
#         --early_stopping 4\
#         --epochs 100\
#         --batch_size 256\
#         --n_flows 6\
#         --seed 42
#   done
# done

# echo "✓ All Pendulum experiments completed."
