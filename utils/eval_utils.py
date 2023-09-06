from collections import defaultdict
from typing import List, Optional
import math
import six
import struct
import numpy as np

def _split_thousands(delimiter, value):
  split = value.split(delimiter)
  return len(split) > 1 and any(map(lambda x: len(x) == 3, split))

def convert_to_float(value):
  """Converts value to a float using a series of increasingly complex heuristics.

  Args:
    value: object that needs to be converted. Allowed types include
      float/int/strings.

  Returns:
    A float interpretation of value.

  Raises:
    ValueError if the float conversion of value fails.
  """
  if isinstance(value, float):
    return value
  if isinstance(value, int):
    return float(value)
  if not isinstance(value, six.string_types):
    raise ValueError("Argument value is not a string. Can't parse it as float")
  sanitized = value

  try:
    # Example: 1,000.7
    if "." in sanitized and "," in sanitized:
      return float(sanitized.replace(",", ""))
    # 1,000
    if "," in sanitized and _split_thousands(",", sanitized):
      return float(sanitized.replace(",", ""))
    # 5,5556
    if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        ",", sanitized):
      return float(sanitized.replace(",", "."))
    # 0.0.0.1
    if sanitized.count(".") > 1:
      return float(sanitized.replace(".", ""))
    # 0,0,0,1
    if sanitized.count(",") > 1:
      return float(sanitized.replace(",", ""))
    return float(sanitized)
  except ValueError:
    # Avoid adding the sanitized value in the error message.
    raise ValueError("Unable to convert value to float")

def _safe_convert_to_float(value):
  float_value = convert_to_float(value)
  if math.isnan(float_value):
    raise ValueError('Value is NaN %s' % value)
  return float_value

def _parse_value(value):
  """Parses a cell value to a number or lowercased string."""
  try:
    return _safe_convert_to_float(value)
  except ValueError:
    try:
      return value.lower()
    except ValueError:
      return value

def _collect_cells_from_table(cell_coos,
                              table):
  cell_values = []
  for cell in cell_coos:
    value = str(table.iat[cell[0], cell[1]])
    cell_values.append(value)
  return cell_values

def execute(aggregation_type, cell_coos,
            table):
  """Executes predicted structure against a table to produce the denotation."""
  values = _collect_cells_from_table(cell_coos, table)
  values_parsed = [_parse_value(value) for value in values]
  values_parsed = tuple(values_parsed)
  if aggregation_type == "NONE":
    # In this case there is no aggregation
    return values_parsed, values
  else:  # Should perform aggregation.
    if not values and (aggregation_type == "AVERAGE" or
                       aggregation_type == "SUM"):
      # Summing or averaging an empty set results in an empty set.
      # NB: SQL returns null for sum over an empty set.
      return tuple(), values
    if aggregation_type == "COUNT":
      denotation = len(values)
    else:
      # In this case all values must be numbers (to be summed or averaged).
      try:
        values_num = [convert_to_float(value) for value in values]
      except ValueError:
        return values_parsed, values
      if aggregation_type == "SUM":
        denotation = sum(values_num)
      elif aggregation_type == "AVERAGE":
        denotation = sum(values_num) / len(values_num)
      else:
        raise ValueError('Unknwon aggregation type: %s' % aggregation_type)
    return tuple([float(denotation)]), values

def to_float32(v):
  """If v is a float reduce precision to that of a 32 bit float."""
  if not isinstance(v, float):
    return v
  return struct.unpack("!f", struct.pack("!f", v))[0]

def to_float32s(elements):
  return tuple(to_float32(v) for v in elements)

def _normalize_float(answer):
  if answer is None:
    return None
  try:
    value = convert_to_float(answer)
    if isinstance(value, float) and math.isnan(value):
      return None
    return value
  except ValueError:
    return answer.lower()


def normalize_answers(answers):
  normalized_answers = (_normalize_float(a) for a in answers)
  normalized_answers = (a for a in normalized_answers if a is not None)
  normalized_answers = (str(a) for a in normalized_answers)
  normalized_answers = list(normalized_answers)
  normalized_answers.sort()
  return normalized_answers


delimiter = ', '
def evaluate_example(predict_str: str, ground_str: str):
    predict_spans = predict_str.split(delimiter)
    ground_spans = ground_str.split(delimiter)
    predict_values = defaultdict(lambda: 0)
    ground_values = defaultdict(lambda: 0)
    for span in predict_spans:
        try:
            predict_values[float(span)] += 1
        except ValueError:
            predict_values[span.strip()] += 1
    for span in ground_spans:
        try:
            ground_values[float(span)] += 1
        except ValueError:
            ground_values[span.strip()] += 1
    _is_correct = predict_values == ground_values
    return _is_correct

def get_denotation_accuracy(predictions: List[str], references: List[str]):
    assert len(predictions) == len(references)
    correct_num = 0
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
        if is_correct:
            correct_num += 1
    return correct_num / len(predictions)

def get_sqa_denotation_accuracy(predictions: List[str], references: List[str]):
    assert len(predictions) == len(references)
    corrects = []
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
        if is_correct:
            corrects.append(1)
        else:
            corrects.append(0)
    return corrects


def calculate_final_scores(output_data):
    orig_accs = {}
    for example in output_data:
        if example["perturbation_type"] == "original":
            orig_accs[example["original_id"]] = example["accuracy"]

    types = ["original", "synonym", "abbreviation", "row", "column", "extend", "masked", "add", "word", "sentence", "combined"]

    accs = {}
    for type in types:
        accs[type] = {
            "original": [],
            "perturbed": [],
            "both_correct": []
        }
        for example in output_data:
            if example["perturbation_type"] == type:
                accs[type]["perturbed"].append(example["accuracy"])
                accs[type]["original"].append(orig_accs[example["original_id"]])
                if example["accuracy"] == 1 and orig_accs[example["original_id"]] == 1:
                    accs[type]["both_correct"].append(1)

    prediction_results = {}
    for type in types:
        if accs[type]["original"] == []:
            continue
        prediction_results[type] = {
            "original_acc": round(np.mean(accs[type]["original"])*100, 4),
            "perturbed_acc": round(np.mean(accs[type]["perturbed"])*100, 4),
            "robust_acc": round(sum(accs[type]["both_correct"]) / sum(accs[type]["original"])*100, 1),
            "num_examples": len(accs[type]["original"]),
        }

    return prediction_results

def calculate_sqa_final_scores(output_data):
    orig_accs = {}
    for example in output_data:
        if example["perturbation_type"] == "original":
            orig_accs[example["original_id"]] = example["accuracy"]

    types = ["original", "synonym", "abbreviation", "row", "column", "extend", "masked", "add", "word", "sentence", "combined"]

    accs = {}
    for type in types:
        accs[type] = {
            "original": [],
            "perturbed": [],
            "both_correct": []
        }
        for example in output_data:
            if example["perturbation_type"] == type:
                accs[type]["perturbed"].append(np.mean(example["accuracy"]))
                accs[type]["original"].append(np.mean(orig_accs[example["original_id"]]))
                both_correct = []
                for acc_1, acc_2 in zip(example["accuracy"], orig_accs[example["original_id"]]):
                    if acc_1 == 1 and acc_2 == 1:
                        both_correct.append(1)
                    else:
                        both_correct.append(0)
                accs[type]["both_correct"].append(np.mean(both_correct))

    prediction_results = {}
    for type in types:
        if accs[type]["original"] == []:
            continue
        prediction_results[type] = {
            "original_acc": round(np.mean(accs[type]["original"])*100, 4),
            "perturbed_acc": round(np.mean(accs[type]["perturbed"])*100, 4),
            "robust_acc": round(sum(accs[type]["both_correct"]) / sum(accs[type]["original"])*100, 1),
            "num_examples": len(accs[type]["original"]),
        }

    return prediction_results