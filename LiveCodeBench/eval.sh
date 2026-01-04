# pip install hf_transfer
# pip install --upgrade "datasets==2.19.1"

model_name=$1 #/mnt/hdfs/tiktok_aiic_new/user/zhengxiaosen.zxs/verl_rl_checkpoints/opencoder_8b_stage1_hit0_only_1/ckptstep_180000/verl-sft_OpenThoughts3_hf_patch/global_step_1530
python -m lcb_runner.runner.main --model ${model_name} --scenario codegeneration --evaluate --release_version release_v6