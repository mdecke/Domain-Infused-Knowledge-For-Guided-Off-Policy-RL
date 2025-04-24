set -e          # abort on first error
set -u          # error on undefined vars

ENV_NAME="Ant-v4"
DATA_DIR="../../data/ant"
BASE_OUT="../../outputs/ant"

MODEL_TYPES=('mle' 'cnf')
INPUT_TYPES=('state' 'state_action' 'prev_state_action' 'state_prev_state')

for MODEL in "${MODEL_TYPES[@]}"; do
  for INPUT in "${INPUT_TYPES[@]}"; do
    OUT_DIR="${BASE_OUT}/${MODEL}/${INPUT}"
    mkdir -p "${OUT_DIR}"

    echo "▶ ${ENV_NAME} | model=${MODEL} | input=${INPUT}"
    python3 -m gop_rl.experiments.fit_model \
        --env_name     "${ENV_NAME}" \
        --model_type   "${MODEL}" \
        --input_type   "${INPUT}" \
        --action_limit 1.0 \
        --data_dir     "${DATA_DIR}" \
        --output_dir   "${OUT_DIR}"
  done
done

echo "✓ All Ant experiments completed."


# # Run Pendulum experiment
# echo "Starting Pendulum experiment..."
# python3 -m gop_rl.experiments.run_rl \
#   --env Pendulum-v1 \
#   --training_steps 20000 \
#   --max_episode_length 200 \
#   --n_grad_steps 3 \
#   --batch_size 1024 \
#   --warm_up 40 \
#   --num_envs 100 \
#   --device 'cuda' \
#   --n_cycles 5 \
#   --eval_freq 2000 \
#   --data_dir ../../data/pendulum \
#   --output_dir ../../outputs/pendulum

# # Check if the second experiment completed successfully
# if [ $? -eq 0 ]; then
#   echo "Pendulum experiment completed successfully."
# else
#   echo "Pendulum experiment failed with exit code $?."
# fi

# echo "All experiments have finished."