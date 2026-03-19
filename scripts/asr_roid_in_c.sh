gpuid=0
method_list=(roid)

prior_correction=False
asr_on=True

for method in ${method_list[@]} ; do
    CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/in_c/${method}.yaml \
                                                            CORRUPTION.NUM_EX 5000 \
                                                            PRINT_EVERY 1 \
                                                            ROID.USE_PRIOR_CORRECTION ${prior_correction} \
                                                            ASR.ON ${asr_on}
done
