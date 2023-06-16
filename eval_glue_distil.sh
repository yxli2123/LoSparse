# MNLI
# Trained on a single V100 32GB
python run_glue_distil.py \
--task_name mnli \
--model_name_or_path bert-base-uncased  \
--stored_model_path pruned_student_model \
--eval_checkpoint eval_model \
--teacher_path textattack/bert-base-uncased-MNLI \
--num_train_epochs 0 \
--output_dir output \
--low_rank_parameter_ratio 0.05 \

