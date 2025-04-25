#!/bin/bash

# Run Ant experiment
echo "Starting Ant experiment..."
python3 -m gop_rl.experiments.run_rl \
  --env Ant-v4 \
  --training_steps 20000 \
  --max_episode_length 1000 \
  --n_grad_steps 3 \
  --batch_size 1024 \
  --warm_up 40 \
  --num_envs 100 \
  --device 'cuda' \
  --n_cycles 1 \
  --eval_freq 500 \
  --data_dir ../../data/ant \
  --output_dir ../../outputs/ant

# Check if the first experiment completed successfully
if [ $? -eq 0 ]; then
  echo "Ant experiment completed successfully."
else
  echo "Ant experiment failed with exit code $?."
fi

# Run Pendulum experiment
echo "Starting Pendulum experiment..."
python3 -m gop_rl.experiments.run_rl \
  --env Pendulum-v1 \
  --training_steps 20000 \
  --max_episode_length 200 \
  --n_grad_steps 3 \
  --batch_size 1024 \
  --warm_up 40 \
  --num_envs 100 \
  --device 'cuda' \
  --n_cycles 5 \
  --eval_freq 2000 \
  --data_dir ../../data/pendulum \
  --output_dir ../../outputs/pendulum

# Check if the second experiment completed successfully
if [ $? -eq 0 ]; then
  echo "Pendulum experiment completed successfully."
else
  echo "Pendulum experiment failed with exit code $?."
fi

echo "All experiments have finished."