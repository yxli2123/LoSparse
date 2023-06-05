# XSum
# Trained on a single V100 32GB
python run_summarization.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name xsum \
  --dataset_config "3.0.0" \
  --output_dir output/xsum/lrr0.35_pr0.05/lr6e-5/seed_42 \
  --max_source_length 256 \
  --seed 42 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --deltaT 100 \
  --low_rank_parameter_ratio 0.35 \
  --final_threshold 0.05 \
  --learning_rate 6e-5 \
  --initial_warmup 2 \
  --final_warmup 8 \
  --num_train_epochs 12 \
  --with_tracking \
  --checkpointing_steps epoch \
  --beta1 0.85 \
  --num_beams 6

# CNN/DailyMail
# Trained on a single V100 32GB
python run_summarization.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name cnn_dailymail \
  --dataset_config "3.0.0" \
  --output_dir output/xsum/lrr0.35_pr0.05/lr6e-5/seed_42 \
  --max_source_length 256 \
  --seed 42 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --deltaT 100 \
  --low_rank_parameter_ratio 0.35 \
  --final_threshold 0.05 \
  --learning_rate 6e-5 \
  --initial_warmup 2 \
  --final_warmup 8 \
  --num_train_epochs 12 \
  --with_tracking \
  --checkpointing_steps epoch \
  --beta1 0.85 \
  --num_beams 6