
cd /home/notebook/code/personal/S9049747/projects/BasicSR

export EXP_ID=0915-0202

git worktree add ../BasicSR_$EXP_ID $EXP_ID
cd ../BasicSR_$EXP_ID

sudo apt update
sudo apt install -y libgl1-mesa-glx
pip3 install -r requirements.txt
BASICSR_EXT=True && python3 setup.py develop

git clone https://github.com/xtudbxk/shared_memory38.git
cd shared_memory38 && python3 setup.py install
cd ..

# ./scripts/dist_train.sh 2 options/train/EDVR/0915-0202.yml
python basicsr/train.py -opt options/train/EDVR/0915-0202.yml
