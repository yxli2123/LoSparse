# MNLI
# Trained on a single V100 32GB
python run_glue.py \
--task_name mnli \
--model_name_or_path bert-base-uncased  \
--per_device_eval_batch_size 16 \
--num_train_epochs 0 \
--output_dir output \
--low_rank_parameter_ratio 0.05 \
--eval_checkpoint eval_checkpoint \
