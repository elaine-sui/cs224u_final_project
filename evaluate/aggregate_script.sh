# AGGREGATION_TYPES=("baseline" "direction" "forward_negation" "backward_negation" "forward_randomized_order" "backward_randomized_order" "forward_all" "backward_all" "all")
# MERGE_ANSWER_TYPES=('hard' 'soft')
# MERGE_COT_TYPES=('intersection' 'union' 'longest' 'majority' 'none')
# PATH_SELECTION_TYPES=('longest' 'shortest' 'heaviest' 'none')

# for agg_type in "${AGGREGATION_TYPES[@]}"; do
#     for merge_answer_type in "${MERGE_ANSWER_TYPES[@]}"; do
#         for merge_cot_type in "${MERGE_COT_TYPES[@]}"; do
#             for path_selection_type in "${PATH_SELECTION_TYPES[@]}"; do
#                 python3 evaluate/aggregate_results.py \
#                     --aggregation_type $agg_type \
#                     --merge_answer_type $merge_answer_type \
#                     --merge_cot_type $merge_cot_type \
#                     --path_selection $path_selection_type
#             done
#         done
#     done
# done

AGGREGATION_TYPES=("baseline")
MERGE_ANSWER_TYPES=('hard')
MERGE_COT_TYPES=('intersection' 'union' 'longest' 'majority' 'none')
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