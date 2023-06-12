# MNLI
# Trained on a single V100 32GB
python run_glue.py \
--task_name mnli \
--model_name_or_path microsoft/deberta-v3-base  \
--warmup_steps 675 \
--initial_threshold 1 \
--final_threshold 0.15 \
--initial_warmup 1 \
--final_warmup 5 \
--beta1 0.85 \
--per_device_train_batch_size 32 \
--learning_rate 9e-5 \
--num_train_epochs 8 \
--output_dir output \
--num_warmup_steps 480 \
--low_rank_parameter_ratio 0.05 \
--seed 7 \
--gradient_accumulation_steps 8
