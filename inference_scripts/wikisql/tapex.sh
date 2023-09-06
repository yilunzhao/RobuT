export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$(pwd)
dataset_name="yilunzhao/robut"
split_name="wikisql"
model_name="microsoft/tapex-large-finetuned-wikisql"

python main/run_tapex_omnitab.py \
  --do_predict \
  --output_dir outputs/${split_name} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 128 \
  --split_name ${split_name} \
  --dataset_name ${dataset_name} \
  --per_device_eval_batch_size 48 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5