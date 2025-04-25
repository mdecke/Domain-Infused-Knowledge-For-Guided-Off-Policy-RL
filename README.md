# Domain-Infused-Knowledge-For-Guided-Off-Policy-RL
Repository for Master's Thesis conducted at ETH Zurich and Caltech


python3 -m gop_rl.experiments.fit_model --env_name Ant-v4 --batch_size 256 --action_limit 1.0 --output_dir outputs/ant --data_dir data/ant --standardize True --input_type 'state'

python3 -m gop_rl.experiments.get_expert_data --num_episodes 5000
