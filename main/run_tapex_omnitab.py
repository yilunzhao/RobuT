#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The Microsoft and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for tapex on table-based question answering tasks.
Adapted from script: https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
"""

import logging
import os
import sys
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
import json
import random
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TapexTokenizer,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from utils.eval_utils import *


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pretrained tokenizer name or path if not the same as model_name. "
                "By default we use BART-large tokenizer for TAPEX-large."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    split_name: str = field(
        metadata={"help": "The name of the split to use."},
    )
    dataset_name: str = field(
        default="yilunzhao/robut",
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    set_seed(training_args.seed)

    datasets = load_dataset(data_args.dataset_name, split=data_args.split_name)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = 1024
    config.early_stopping = False
    padding = "max_length" if data_args.pad_to_max_length else False

    # load tapex tokenizer
    tokenizer = TapexTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )

    # load Bart based Tapex model (default tapex-large)
    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    column_names = datasets.column_names

    def preprocess_tableqa_function(examples):
        """
        The is_training FLAG is used to identify if we could use the supervision
        to truncate the table content if it is required.
        """

        questions = [question.lower() for question in examples["question"]]
        example_tables = [table for table in examples["table"]]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]

        answers = examples["answers"]

        model_inputs = tokenizer(
            table=tables, query=questions, max_length=data_args.max_source_length, padding=padding, truncation=True
        )

        labels = tokenizer(
            answer=[", ".join(answer) for answer in answers],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_sqa_function(examples, is_training=False):
        """
        The is_training FLAG is used to identify if we could use the supervision
        to truncate the table content if it is required.
        """
        questions = []
        for question in examples["question"]:
            question = ast.literal_eval(question)
            history = ""
            for q in question:
                history += q.lower() + " "
                questions.append(history.strip())

        tables, answers = [], []
        for i, (question, table, answer) in enumerate(zip(examples["question"], examples["table"], examples["answers"])):
            question = ast.literal_eval(question)
            answer = [ast.literal_eval(a) for a in answer]
            for j, q in enumerate(question):
                table_df = pd.DataFrame.from_records(table["rows"], columns=table["header"])
                tables.append(table_df)
                answers.append(answer[j])

        model_inputs = tokenizer(
            table=tables, query=questions, max_length=data_args.max_source_length, padding=padding, truncation=True
        )

        labels = tokenizer(
            answer=[", ".join(answer) for answer in answers],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    if training_args.do_predict:
        max_target_length = data_args.max_target_length
        predict_dataset = datasets
        raw_predict_dataset = datasets

        predict_dataset = predict_dataset.map(
            preprocess_tableqa_function if data_args.split_name in ["wikisql", "wtq"] else preprocess_sqa_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        delimiter = ", "
        accuracy = get_denotation_accuracy(decoded_preds, decoded_labels)
        result = {"denotation_accuracy": accuracy}

        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset= None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)

        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        
        outputs = []
        if data_args.split_name in ["wikisql", "wtq"]:
            for i, pred in enumerate(predictions):
                example = raw_predict_dataset[i]
                answer = ", ".join([i.lower() for i in example["answers"]])
                output_example = example
                output_example["prediction"] = pred
                
                output_example["accuracy"] = 1 if get_denotation_accuracy([pred], [answer]) else 0
                outputs.append(output_example)
        elif data_args.split_name == "sqa":
            for example in raw_predict_dataset:
                preds, answers = [], []
                output_example = example
                output_example["prediction"] = []
                for ref in example["answers"]:
                    ref = ast.literal_eval(ref)
                    pred = predictions.pop(0)
                    answer = ", ".join([i.lower() for i in ref])
                    output_example["prediction"].append(pred)

                    preds.append(pred)
                    answers.append(answer)            
                output_example["accuracy"] = get_sqa_denotation_accuracy(preds, answers)
                outputs.append(output_example)


        
        model_name = "tapex" if "tapex" in model_args.model_name_or_path else "omnitab"
        output_prediction_file = f"{training_args.output_dir}/{model_name}-preds.json"
        json.dump(outputs, open(output_prediction_file, "w"), indent=4)

        if data_args.split_name == "sqa":
            prediction_results = calculate_sqa_final_scores(outputs)
        else:
            prediction_results = calculate_final_scores(outputs)
        output_prediction_file = f"{training_args.output_dir}/{model_name}-scores.json"
        json.dump(prediction_results, open(output_prediction_file, "w"), indent=4)






def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

    