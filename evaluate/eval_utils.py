import os
import pickle
from glob import glob
import pandas as pd
import re
from pathlib import Path

import metrics

import sys
sys.path.append("../")
from utils import get_all_cot_steps


def restrict_to_not_only(output_df):
    new_df = []
    for _, row in output_df.iterrows():
        query = row['test_example']['query'].split(":")[-1].strip()
        if "not" in query:
            new_df.append(row.to_dict())
    
    new_df = pd.DataFrame(new_df)

    return new_df

def restrict_to_no_not_only(output_df):
    new_df = []
    for _, row in output_df.iterrows():
        query = row['test_example']['query'].split(":")[-1].strip()
        if "not" not in query:
            new_df.append(row.to_dict())
    
    new_df = pd.DataFrame(new_df)

    return new_df


def get_simple_eval_metrics(path, restrict_type='none'):
    with open(path, 'rb') as f:
        output_df = pickle.load(f)

    if restrict_type == "not_only":
        output_df = restrict_to_not_only(output_df)
    elif restrict_type == "no_not_only":
        output_df = restrict_to_no_not_only(output_df)

    output_df['label_acc'] = output_df.apply(lambda x: metrics.strict_acc(x['predicted_answer'], x['gold_answer']), axis=1)
    output_df['cot_acc'] = output_df.apply(lambda x: metrics.strict_acc(x['predicted_cot'], x['gold_cot']), axis=1)
    output_df['cot_precision'] = output_df.apply(lambda x: metrics.cot_precision(get_all_cot_steps([x['predicted_cot']]), get_all_cot_steps([x['gold_cot']])), axis=1)
    output_df['cot_recall'] = output_df.apply(lambda x: metrics.cot_recall(get_all_cot_steps([x['predicted_cot']]), get_all_cot_steps([x['gold_cot']])), axis=1)
    output_df['cot_f1'] = output_df.apply(lambda x: metrics.f1_score(x['cot_precision'], x['cot_recall']), axis=1)

    avg_acc = output_df['label_acc'].values.mean()
    average_cot_acc = output_df['cot_acc'].values.mean()
    avg_cot_precision = output_df['cot_precision'].values.mean()
    avg_cot_recall = output_df['cot_recall'].values.mean()
    avg_cot_f1 = output_df['cot_f1'].values.mean()

    return {
        "avg_label_acc": avg_acc,
        "avg_cot_acc": average_cot_acc,
        "avg_cot_precision": avg_cot_precision,
        "avg_cot_recall": avg_cot_recall,
        "avg_cot_f1": avg_cot_f1
    }

def get_metrics_dict_for_path_selection_type(path_glob, path_selection, restrict_type):
    metrics_dict = {} # {aggregation_type: {merge_cot_type: metrics_dict} }
    aggregation_types = []
    merge_cot_types = []
    for filename in glob(path_glob):
        # print(p)
        if "summary" in filename:
            aggregation_type = Path(filename).parts[-3]
        else:
            aggregation_type = Path(filename).parts[-2]

        if aggregation_type not in aggregation_types:
            aggregation_types.append(aggregation_type)
            metrics_dict[aggregation_type] = {}

        merge_cot_type = re.findall(r'merge_cot_(.*)_path', filename)[0]
        if merge_cot_type == 'none' and path_selection != "heaviest":
            continue
        
        if merge_cot_type not in merge_cot_types:
            merge_cot_types.append(merge_cot_type)
        
        # print(merge_cot_type)

        metrics_dict[aggregation_type][merge_cot_type] = get_simple_eval_metrics(filename, restrict_type=restrict_type)
    return metrics_dict, aggregation_types, merge_cot_types

def get_data_for_metric(metric, metrics_dict, aggregation_types, merge_cot_types):
    data = []
    for aggregation_type in aggregation_types:
        data_aggregation = [aggregation_type]
        for merge_cot_type in merge_cot_types:
            data_aggregation.append(metrics_dict[aggregation_type][merge_cot_type][metric])
        
        data.append(data_aggregation)
    return data

def get_best_val_per_merge_type(data_df, merge_cot_types):
    print('='*80)
    metrics = {}
    for column in merge_cot_types:
        metrics[column] = data_df[column].values.max()
    
    # print in sorted order
    metrics = sorted(metrics.items(), key=lambda x:x[1], reverse=True)
    for k, v in metrics:
        print(f"{k} max: {v}")

def get_best_val_per_aggregation_type(data_df, aggregation_types, merge_cot_types):
    print('='*80)
    metrics = {}
    for aggregation_type in aggregation_types:
        max_val = data_df[data_df['Aggregation Type'] == aggregation_type][merge_cot_types].values.max()
        metrics[aggregation_type] = max_val
    
    # print in sorted order
    metrics = sorted(metrics.items(), key=lambda x:x[1], reverse=True)
    for k, v in metrics:
        print(f"{k} max: {v}")

def plot_bar_chart(metric, columns, metrics_dict, aggregation_types, merge_cot_types, title=None):
    data = get_data_for_metric(metric, metrics_dict, aggregation_types, merge_cot_types)
    # print(data)
    metrics_df = pd.DataFrame(
            columns=columns,
            data=data
    )

    get_best_val_per_merge_type(metrics_df, merge_cot_types)
    get_best_val_per_aggregation_type(metrics_df, aggregation_types, merge_cot_types)

    # plot grouped bar chart
    if title is None:
        title = metric
    metrics_df.plot(x='Aggregation Type',
            kind='bar',
            stacked=False,
            title=title
    )