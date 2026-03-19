gpuid=0
method_list=(eta)
seed_list=(0 1 2 3 4 5 6 7 8 9)

asr_on=True

for method in ${method_list[@]} ; do
    for seed in ${seed_list[@]} ; do
        CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/cin_c/${method}/${method}${seed}.yaml \
                                                                PRINT_EVERY 1 \
                                                                CORRUPTION.NUM_EX 50000 \
                                                                ASR.ON ${asr_on}
    done
done
