for seed in 1234 5678; do
    python3 run.py \
        --prompt_type backward \
        --randomized_order \
        --seed $seed
done