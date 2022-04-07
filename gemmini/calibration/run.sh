for i in {0..7}; do
    #mkdir -p output_dir_${i}
    mkdir -p output_dir_${i}_imperfect
    # timeloop-model ../arch/* ../prob/mm/mm_${i}.yaml mapspace/mapping${i}.yaml -o output_dir_${i}
    timeloop-model ../arch/* ../prob/mm/mm_${i}.yaml mapspace/mapping${i}_imperfect.yaml -o output_dir_${i}_imperfect
done


# for i in {0..7}; do
for i in {0..7}; do
    # echo "mm_${i} perfect factorization"
    # grep 'Cycles: ' output_dir_${i}/timeloop-model.stats.txt 
    # grep 'Energy: ' output_dir_${i}/timeloop-model.stats.txt
    echo "mm_${i} imperfect factorization"
    grep 'Cycles: ' output_dir_${i}_imperfect/timeloop-model.stats.txt 
    grep 'Energy: ' output_dir_${i}_imperfect/timeloop-model.stats.txt

done
