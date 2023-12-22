mkdir dataset_vm
# mount -t tmpfs tmpfs ./dataset_vm
sudo mount -t ramfs none ./dataset_vm
free -m

sudo chmod 777 dataset_vm
echo 'start to copy train_sharp_bicubic into ramfs'
cp /home/notebook/code/personal/S9049747/zhengqiangzhang/REDS/train/train_sharp_bicubic ./dataset_vm/ -r
free -m

echo 'start to copy train_sharp into ramfs'
cp /home/notebook/code/personal/S9049747/zhengqiangzhang/REDS/train/train_sharp ./dataset_vm/ -r
free -m

cd dataset_vm
find  /home/notebook/code/personal/S9049747/zhengqiangzhang/REDS/train/train_sharp -type d |cut -d '/' -f 10- |xargs -l sudo chmod 777
find  /home/notebook/code/personal/S9049747/zhengqiangzhang/REDS/train/train_sharp_bicubic -type d |cut -d '/' -f 10- |grep -v '.lmdb'|xargs -l sudo chmod 777

free -m
