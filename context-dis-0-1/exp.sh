#!/bin/bash

set -e

PREFIX=$1
LOCAL_PREFIX=exp0309g3
LOCAL_MODEL_NAME=local_4900_6
SAVE_PREDICTION_FORMAT=$2
FEATURE_COHERENCE_TYPE=$3

SKIP_LOCAL=TRUE
SAVE_PREDICTION_TOP_K=3
START_GLOBAL_ITERATION=2
MAX_GLOBAL_ITERATION=5

TRAIN_DATASET='aida_conll'
TRAIN_ID_START=371182
TRAIN_ID_END=394577
TEST_ID_END=406110
TRAIN_ID_STEP=4000
TEST_ID_STEP=4000

TRAIN_ITERATIONS=9000
TRAIN_MAX_DEPTH=8

OTHER_TEST_DATASETS='msnbc_new aquaint_new ace2004_wned'
OTHER_TEST_ID_START=0
OTHER_TEST_ID_END=12000
OTHER_TEST_ID_STEP=4000

TRAIN_ID_SPLITS=$(seq $TRAIN_ID_START $TRAIN_ID_STEP $TRAIN_ID_END)
TEST_ID_SPLITS=$(seq $TRAIN_ID_END $TEST_ID_STEP $TEST_ID_END)
OTHER_TEST_ID_SPLITS=$(seq $OTHER_TEST_ID_START $OTHER_TEST_ID_STEP $OTHER_TEST_ID_END)

if [ $SKIP_LOCAL != 'TRUE' ] ; then
  ### Generate Local Features
  for i in $TRAIN_ID_SPLITS ; do
    python fea_gen_kp_redis.py $i $((i+TRAIN_ID_STEP-1)) "${TRAIN_DATASET}" train debug "${PREFIX}:fea_ctx" ctx &
    python fea_gen_kp_redis.py $i $((i+TRAIN_ID_STEP-1)) "${TRAIN_DATASET}" train debug "${PREFIX}:fea_coref" coref &
  done
  for t in testa testb ; do
    for i in $TEST_ID_SPLITS ; do
      python fea_gen_kp_redis.py $i $((i+TEST_ID_STEP-1)) "${TRAIN_DATASET}" $t debug "${PREFIX}:fea_ctx" ctx &
      python fea_gen_kp_redis.py $i $((i+TEST_ID_STEP-1)) "${TRAIN_DATASET}" $t debug "${PREFIX}:fea_coref" coref &
    done
  done
  wait
  for d in ${OTHER_TEST_DATASETS} ; do
    for i in $OTHER_TEST_ID_SPLITS ; do
      python fea_gen_kp_redis.py $i $((i+OTHER_TEST_ID_STEP-1)) $d test debug "${PREFIX}:fea_ctx" ctx &
      python fea_gen_kp_redis.py $i $((i+OTHER_TEST_ID_STEP-1)) $d test debug "${PREFIX}:fea_coref" coref &
    done
  done
  wait

  ### Train Local Model
  python process.py train_local "${TRAIN_DATASET}" train -p "${PREFIX}" -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"

  ### Test Local Model
  python process.py test_local "${TRAIN_DATASET}" testa -p "${PREFIX}" -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  python process.py test_local "${TRAIN_DATASET}" testb -p "${PREFIX}" -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  for d in ${OTHER_TEST_DATASETS} ; do
    python process.py test_local $d test -p "${PREFIX}" -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  done
fi

for it in $(seq $START_GLOBAL_ITERATION $MAX_GLOBAL_ITERATION) ; do
  ### Save Predictions
  for t in train testa testb ; do
    python process.py save_predictions "${TRAIN_DATASET}" $t -p "${PREFIX}" -l "${LOCAL_PREFIX}" -L "${LOCAL_MODEL_NAME}" -c ${FEATURE_COHERENCE_TYPE} -g $((it-1)) -s $SAVE_PREDICTION_FORMAT -t $SAVE_PREDICTION_TOP_K -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  done
  for d in ${OTHER_TEST_DATASETS} ; do
    python process.py save_predictions $d test -p "${PREFIX}" -l "${LOCAL_PREFIX}" -L "${LOCAL_MODEL_NAME}" -c ${FEATURE_COHERENCE_TYPE} -g $((it-1)) -s $SAVE_PREDICTION_FORMAT -t $SAVE_PREDICTION_TOP_K -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  done

  ### Generate Global Features
  for i in $TRAIN_ID_SPLITS ; do
    python fea_gen_kp_redis.py $i $((i+TRAIN_ID_STEP-1)) "${TRAIN_DATASET}" train debug "${PREFIX}:fea_${FEATURE_COHERENCE_TYPE}${it}" $FEATURE_COHERENCE_TYPE &
  done 
  for t in testa testb ; do
    for i in $TEST_ID_SPLITS ; do
      python fea_gen_kp_redis.py $i $((i+TEST_ID_STEP-1)) "${TRAIN_DATASET}" $t debug "${PREFIX}:fea_${FEATURE_COHERENCE_TYPE}${it}" $FEATURE_COHERENCE_TYPE &
    done
  done
  for d in ${OTHER_TEST_DATASETS} ; do
    for i in $OTHER_TEST_ID_SPLITS ; do
      python fea_gen_kp_redis.py $i $((i+OTHER_TEST_ID_STEP-1)) $d test debug "${PREFIX}:fea_${FEATURE_COHERENCE_TYPE}${it}" $FEATURE_COHERENCE_TYPE &
    done
  done
  wait

  ### Train Global Model
  python process.py train_global "${TRAIN_DATASET}" train -p "${PREFIX}" -l "${LOCAL_PREFIX}" -c ${FEATURE_COHERENCE_TYPE} -g $it -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"

  ### Test Global Model
  python process.py test_global "${TRAIN_DATASET}" testa -p "${PREFIX}" -l "${LOCAL_PREFIX}" -c ${FEATURE_COHERENCE_TYPE} -g $it -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  python process.py test_global "${TRAIN_DATASET}" testb -p "${PREFIX}" -l "${LOCAL_PREFIX}" -c ${FEATURE_COHERENCE_TYPE} -g $it -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  for d in ${OTHER_TEST_DATASETS} ; do
    python process.py test_global $d test -p "${PREFIX}" -l "${LOCAL_PREFIX}" -c ${FEATURE_COHERENCE_TYPE} -g $it -i ${TRAIN_ITERATIONS} -d ${TRAIN_MAX_DEPTH} 2>&1 | tee -a "${PREFIX}.${HOSTNAME}.log"
  done
done

