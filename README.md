# RobuT
Data and code for ACL 2023 paper "RobuT: A Systematic Study of Table QA Robustness Against Human-Annotated Adversarial Perturbations"

## Prepare Environment
We officially support python 3.9. You could use following commands to install the required packages
```
pip install -r requirements.txt
```

## RobuT Dataset
We have released our dataset on [HuggingFace](https://huggingface.co/datasets/yilunzhao). Use the following command to load the dataset (we use `RobuT-WTQ` as an example):
```python
datasets = load_dataset("yilunzhao/robut", split="wtq")
```
The dataset can also be found in `robut_data.zip` file.

## Experiments
To run each model on each RobuT subset, please execute the inference scripts in the `inference_scripts` directory. For example, use the following command to evaluate the performance of TAPEX on RobuT-WTQ:
```
bash inference_scripts/wtq/tapex.sh
```
The corresponding model output and scores can be found at `outputs/wtq/tapex-preds.json` and `outputs/wtq/tapex-scores.json`, respectively.

## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu).

## Citation
```
@inproceedings{zhao-etal-2023-robut,
    title = "{R}obu{T}: A Systematic Study of Table {QA} Robustness Against Human-Annotated Adversarial Perturbations",
    author = "Zhao, Yilun  and
      Zhao, Chen  and
      Nan, Linyong  and
      Qi, Zhenting  and
      Zhang, Wenlin  and
      Tang, Xiangru  and
      Mi, Boyu  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.334",
    doi = "10.18653/v1/2023.acl-long.334",
    pages = "6064--6081",
}
```