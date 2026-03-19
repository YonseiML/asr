gpuid=0
method_list=(eta)

asr_on=True

for method in ${method_list[@]} ; do
    CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/in_c/${method}.yaml \
                                                            CORRUPTION.NUM_EX 5000 \
                                                            PRINT_EVERY 1 \
                                                            ASR.ON ${asr_on}
done
