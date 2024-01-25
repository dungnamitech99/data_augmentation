
factor=1.05
suffix="_$factor"
in_list=data/fa/fa_nami_1/label_time_wav.txt
out_list=data/fa/fa_nami_1/label_time_wav${suffix}.txt
out_wav_dir=/disk2/projects/Data/WakeWord_DATA/release/train/ftel_train/negative_3s_16k/wav${suffix}

python utils/perturb_speed.py -f $factor \
                                -i $in_list \
                                -d $out_wav_dir \
                                -o $out_list