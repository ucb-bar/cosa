RANDOM_SEEDS="1 888 1234 56781234 103495333"
#RANDOM_SEEDS="1"
MODELS="alexnet resnext50_32x4d deepbench"
#MODELS="alexnet"

for RANDOM_SEED in ${RANDOM_SEEDS}; do
    for MODEL in ${MODELS}; do
        python bo.py --output_dir output_dir --arch_dir arch_bo --random_seed ${RANDOM_SEED} --model ${MODEL} --num_samples 2100 &
    done
done
