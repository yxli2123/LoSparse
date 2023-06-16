# MNLI
# Trained on a single V100 32GB
python run_glue_distil.py \
--task_name mnli \
--model_name_or_path bert-base-uncased  \
--stored_model_path pruned_student_model \
--teacher_path textattack/bert-base-uncased-MNLI \
--per_device_train_batch_size 32 \
--learning_rate 9e-5 \
--alpha_output 0 \
--alpha_layer 15 \
--num_train_epochs 50 \
--output_dir output \
--low_rank_parameter_ratio 0.05 \
--seed 7 \

