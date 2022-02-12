#!/bin/bash

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM RDCnet training job $SLURM_JOB_ID"
echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"


CONDAENV=/tungstenfs/scratch/gmicro_share/_software/CondaEnvs/Linux/FDE

EXPPATH=/tungstenfs/scratch/gmicro/buchtimo/gitrepos/FourierImageTransformer/examples/trec_kanji_sum

# Work code - ISIT
${CONDAENV}/bin/python ${EXPPATH}/train_trec_kanji.py --config ${EXPPATH}/trec_kanji_train.conf
EXITCODE=$?


END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow finished with code $EXITCODE"
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
