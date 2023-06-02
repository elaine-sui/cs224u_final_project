# for seed in 1234 5678 910; do
#     python3 run.py \
#         --prompt_type baseline \
#         --temperature 0.7 \
#         --seed $seed
# done
python3 run.py \
    --prompt_type forward \
    --temperature 0.0 \
    --test_mini_batch \
    --k 1
    