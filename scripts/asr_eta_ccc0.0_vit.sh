gpuid=0
method_list=(eta)
level_list=(0.0)
split_list=(0 1 2 3 4 5 6 7 8)

backbone=vit_b_16
asr_on=True
asr_m0=0.9995
asr_p0=5.0
asr_a0=0.005
asr_r0=0.5
asr_r1=0.1

for method in ${method_list[@]} ; do
    for level in ${level_list[@]} ; do
        for split in ${split_list[@]} ; do
            CUDA_VISIBLE_DEVICES=${gpuid} python test_time.py --cfg cfgs/ccc/${level}/${method}/${method}${split}.yaml \
                                                                    PRINT_EVERY 1 \
                                                                    MODEL.ARCH ${backbone} \
                                                                    ASR.ON ${asr_on} \
                                                                    ASR.M0 ${asr_m0} \
                                                                    ASR.P0 ${asr_p0} \
                                                                    ASR.A0 ${asr_a0} \
                                                                    ASR.R0 ${asr_r0} \
                                                                    ASR.R1 ${asr_r1}
        done
    done
done
