for seed in 1234 5678 910; do
    python3 run.py \
        --prompt_type baseline \
        --temperature 0.7 \
        --seed $seed
done
    