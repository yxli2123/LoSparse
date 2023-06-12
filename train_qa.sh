# Squad
# Trained on a single V100 32GB
CUDA_VISIBLE_DEVICES=6 python run_qa.py \
--dataset_name squad \
--model_name_or_path bert-base-uncased \
--warmup_steps 5400 \
--initial_threshold 1 \
--final_threshold 0.45 \
--initial_warmup 1 \
--final_warmup 5 \
--beta1 0.85 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 10 \
--output_dir losparse_squad/lr5e-5/seed7 \
--num_warmup_steps 3681 \
--low_rank_parameter_ratio 0.05 \
--seed 7 \