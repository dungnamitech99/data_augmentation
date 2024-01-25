docker run -p 6018:6006 --name dunghc_wuw_3 -it -e HOME=/tf --gpus all -u $(id -u):$(id -g) \
    -v $PWD:/tf/train_hiFPToi \
    -v /mnt/sdb/:/mnt/sdb/ \
    tensorflow/tensorflow:2.10.1-gpu-hiFPToi \
    bash 
    
#cd /tf/train_hiFPToi
#python wuw/train.py <conf> <expdir>

