gpuid=0
method_list=(eta)

asr_on=True

for method in ${method_list[@]} ; do
    CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/in_d/${method}.yaml \
                                                            PRINT_EVERY 1 \
                                                            ASR.ON ${asr_on}
done
