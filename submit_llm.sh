#!/bin/bash
#SBATCH --account=PAS2138
#SBATCH --output=/users/PAS2138/roozbehn99/ellm/store/gif_sbatch_ac_%j.log
#SBATCH --error=/users/PAS2138/roozbehn99/ellm/store/train_ac_crafter_%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name=test_llm
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --time=00:30:00


python testLLMs.py
# python train.py use_wandb=true env_spec.use_sbert=True use_goal=True use_language_state=sbert env_spec.lm_spec.lm_class=SimpleOracle exp_name=testing # Training the oracle
# python train.py use_wandb=true agent._target_=baselines.icm_apt.ICMAPTAgent env_spec.use_sbert=True use_goal=True use_language_state=sbert exp_name=TEST_RUN_APT # agent.use_language=true 

