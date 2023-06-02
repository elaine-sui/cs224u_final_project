for seed in 5678 910; do
    python3 run.py \
        --prompt_type backward \
        --randomized_order \
        --seed $seed
done