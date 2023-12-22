
cd /home/notebook/code/personal/S9049747/projects/BasicSR

export EXP_ID=0905-02

git worktree add ../BasicSR_$EXP_ID $EXP_ID
cd ../BasicSR_$EXP_ID

sudo apt update
sudo apt install -y libgl1-mesa-glx
pip3 install -r requirements.txt
BASICSR_EXT=True && python3 setup.py develop
# ./scripts/dist_train.sh 2 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.zzq.yml
python basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS.zzq.yml 
