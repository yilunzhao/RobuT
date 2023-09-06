model_name="google/tapas-large-finetuned-wikisql-supervised"

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$(pwd)

python main/run_tapas.py \
    --model_name ${model_name} \
    --dataset_name yilunzhao/robut \
    --split_name wikisql \
    --device cuda