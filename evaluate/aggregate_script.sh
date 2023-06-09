AGGREGATION_TYPES=("single_baseline" "single_forward" "single_backward" "single_baseline_neg" "single_forward_neg" "single_backward_neg" "double_baseline_negation" "double_forward_negation" "double_backward_negation" "baseline" "direction" "forward_negation" "backward_negation" "forward_randomized_order" "backward_randomized_order" "forward_all" "backward_all" "all")
MERGE_ANSWER_TYPES=('hard')
MERGE_COT_TYPES=('none')
PATH_SELECTION_TYPES=('heaviest')

for agg_type in "${AGGREGATION_TYPES[@]}"; do
    for merge_answer_type in "${MERGE_ANSWER_TYPES[@]}"; do
        for merge_cot_type in "${MERGE_COT_TYPES[@]}"; do
            for path_selection_type in "${PATH_SELECTION_TYPES[@]}"; do
                python3 evaluate/aggregate_results.py \
                    --aggregation_type $agg_type \
                    --merge_answer_type $merge_answer_type \
                    --merge_cot_type $merge_cot_type \
                    --path_selection $path_selection_type
            done
        done
    done
done

for agg_type in "${AGGREGATION_TYPES[@]}"; do
    for merge_answer_type in "${MERGE_ANSWER_TYPES[@]}"; do
        for merge_cot_type in "${MERGE_COT_TYPES[@]}"; do
            for path_selection_type in "${PATH_SELECTION_TYPES[@]}"; do
                python3 evaluate/aggregate_results.py \
                    --aggregation_type $agg_type \
                    --merge_answer_type $merge_answer_type \
                    --merge_cot_type $merge_cot_type \
                    --path_selection $path_selection_type \
                    --merge_cot_of_majority_answer
            done
        done
    done
done

AGGREGATION_TYPES=("double_baseline_negation" "double_forward_negation" "double_backward_negation")
MERGE_ANSWER_TYPES=('hard')
MERGE_COT_TYPES=('none')
PATH_SELECTION_TYPES=('none')

for agg_type in "${AGGREGATION_TYPES[@]}"; do
    for merge_answer_type in "${MERGE_ANSWER_TYPES[@]}"; do
        for merge_cot_type in "${MERGE_COT_TYPES[@]}"; do
            for path_selection_type in "${PATH_SELECTION_TYPES[@]}"; do
                python3 evaluate/aggregate_results.py \
                    --aggregation_type $agg_type \
                    --merge_answer_type $merge_answer_type \
                    --merge_cot_type $merge_cot_type \
                    --path_selection $path_selection_type
            done
        done
    done
done