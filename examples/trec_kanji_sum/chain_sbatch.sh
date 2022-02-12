#!/bin/sh

id=`sbatch --account=dlthings --job-name=fit_kp --cpus-per-task=8 --ntasks=1 --partition=gpu_short --mem=12G --gres=gpu:a40:1 --mail-user=tim-oliver.buchholz@fmi.ch --mail-type=ALL run_training.sh`

[[ "$id" =~ Submitted\ batch\ job\ ([0-9]+) ]]
id="${BASH_REMATCH[1]}"

echo "$id"

for n in {1..11}; do
    id=`sbatch --dependency=afterany:${id} --account=dlthings --job-name=fit_ks --cpus-per-task=8 --ntasks=1 --partition=gpu_short --mem=12G --gres=gpu:a40:1 --mail-user=tim-oliver.buchholz@fmi.ch --mail-type=ALL run_training.sh`

    [[ "$id" =~ Submitted\ batch\ job\ ([0-9]+) ]]
    id="${BASH_REMATCH[1]}"
    echo "$id"
done