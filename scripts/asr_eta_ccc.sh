gpuid=0
method_list=(eta)
level_list=(0.0 20.0 40.0)
split_list=(0 1 2 3 4 5 6 7 8)

asr_on=True

for method in ${method_list[@]} ; do
    for level in ${level_list[@]} ; do
        for split in ${split_list[@]} ; do
            CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/ccc/${level}/${method}/${method}${split}.yaml \
                                                                    PRINT_EVERY 1 \
                                                                    ASR.ON ${asr_on}
        done
    done
done
