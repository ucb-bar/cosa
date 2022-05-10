# for i in {0..7}; do
#     mkdir -p output_dir_T512_${i}
#     # timeloop-model ../arch/* ../prob/mm/mm_${i}.yaml mapspace/mapping${i}.yaml -o output_dir_${i}
#     timeloop-model ../../arch/gemmini_transformer.yaml ../../transformer_512/mm_${i}.yaml mapspace/transformer_512_${i}.yaml -o output_dir_T512_${i}
# done


# # for i in {0..7}; do
# for i in {0..7}; do
#     # echo "mm_${i} perfect factorization"
#     # grep 'Cycles: ' output_dir_${i}/timeloop-model.stats.txt 
BERT_ARCH="bert_512x8"
i=$1

FUSED_DIR="output_${BERT_ARCH}_${i}"
NON_FUSED_DIR="output_${BERT_ARCH}_noFused_${i}"

mkdir -p $FUSED_DIR
mkdir -p $NON_FUSED_DIR
timeloop-model ../../arch/gemmini_transformer.yaml ../../prob/transformer_512/mm_${i}.yaml mapspace/${BERT_ARCH}_${i}.yaml -o ${FUSED_DIR}
timeloop-model ../../arch/arch.yaml ../../prob/transformer_512/mm_${i}.yaml mapspace/${BERT_ARCH}_${i}.yaml -o ${NON_FUSED_DIR}

echo "mm_${i} with BW constraint"
grep 'Cycles: ' ${FUSED_DIR}/timeloop-model.stats.txt
grep 'Energy: ' ${FUSED_DIR}/timeloop-model.stats.txt
echo "mm_${i} without BW constraint"
grep 'Cycles: ' ${NON_FUSED_DIR}/timeloop-model.stats.txt
grep 'Energy: ' ${NON_FUSED_DIR}/timeloop-model.stats.txt





