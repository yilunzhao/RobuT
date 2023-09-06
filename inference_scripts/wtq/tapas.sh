model_name="google/tapas-large-finetuned-wtq"

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$(pwd)

python main/run_tapas.py \
    --model_name ${model_name} \
    --dataset_name yilunzhao/robut \
    --split_name wtq \
    --device cuda