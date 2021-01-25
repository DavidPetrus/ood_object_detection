import subprocess
import datetime
from shutil import copyfile
import time

logdir = '24Jan'

cmd1 = cmd2 = cmd3 = cmd4 = cmd5 = cmd6 = cmd7 = cmd8 = cmd9 = cmd10 = \
cmd11 = cmd12 = cmd13 = cmd14 = cmd15 = cmd16 = cmd17 = cmd18 = cmd19 = cmd20 = \
cmd21 = cmd22 = cmd23 = cmd24 = cmd25 = cmd26 = cmd27 = cmd28 = cmd29 = cmd30 = ""

cmd1 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"1  --max_lvis_load=800  --num_preloads=10"

cmd2 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"2 --max_lvis_load=800  --num_preloads=30"

cmd3 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"3 --max_lvis_load=800  --num_preloads=100"

cmd4 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"4 --max_lvis_load=2000  --num_preloads=30"

cmd5 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"5 --max_lvis_load=2000  --num_preloads=100"

cmd6 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"6 --num_train_cats=50"

cmd7 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"7 --num_train_cats=150 --max_lvis_load=4000"

cmd8 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"8 --num_train_cats=150 --max_lvis_load=4000  --num_preloads=100"

cmd9 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"9 --fpn=False"

cmd10 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"10 --large_qry=False"

cmd11 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"11 --supp_level_offset=0  --fpn=False"

cmd12 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"12 --num_sup=10"

'''cmd13 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"13 --inner_lr=0.01 --steps=3"

cmd14 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"14 --alpha=0.1 --bbox_coeff=50. --supp_level_offset=2"

cmd15 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"15 --bbox_coeff=5."

cmd16 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"16 --supp_level_offset=3"

cmd17 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"17 --meta_lr=0.001 --inner_lr=0.01 --steps=2 --alpha=0.1"

cmd18 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"18 --meta_lr=0.001 --inner_lr=0.01 --steps=1 --alpha=0.1"

cmd19 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"19 --meta_lr=0.001 --inner_lr=0.01 --steps=2 --alpha=0.25"

cmd20 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"20 --meta_lr=0.001 --inner_lr=0.01 --steps=1 --alpha=0.25"

cmd21 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"21 --meta_lr=0.001 --inner_lr=0.001 --steps=2 --alpha=0.25"

cmd22 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"22 --meta_lr=0.001 --inner_lr=0.01 --steps=2 --alpha=0.01"

cmd23 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"23 --meta_lr=0.001 --inner_lr=0.01 --steps=1 --alpha=0.01"

cmd24 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"24 --meta_lr=0.001 --inner_lr=0.001 --steps=2 --alpha=0.01"'''

'''cmd25 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"25 --meta_lr=0.001 --inner_lr=0.01 --steps=2 --gamma=3 --train_mode=True"

cmd26 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"26 --meta_lr=0.001 --inner_lr=0.03 --steps=2 --gamma=3 --train_mode=True"

cmd27 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"27 --meta_lr=0.01 --inner_lr=0.03 --steps=1 --gamma=3 --train_mode=True"

cmd28 = "srun -N1 -p ha -l python3 infer.py --exp="+logdir+"28 --meta_lr=0.001 --inner_lr=0.001 --steps=2 --gamma=3 --train_mode=True"

cmd29 = "srun -N1 -p batch -l python3 infer.py --exp="+logdir+"29 --meta_lr=0.01 --inner_lr=0.001 --steps=2 --gamma=3 --train_mode=True"

cmd30 = "srun -N1 -p stampede -l python3 infer.py --exp="+logdir+"30 --meta_lr=0.001 --inner_lr=0.01 --steps=1 --gamma=3 --train_mode=True"'''




one = subprocess.Popen([cmd1],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
two = subprocess.Popen([cmd2],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
three = subprocess.Popen([cmd3],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
four = subprocess.Popen([cmd4],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
five = subprocess.Popen([cmd5],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
six = subprocess.Popen([cmd6],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
seven = subprocess.Popen([cmd7],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
eight = subprocess.Popen([cmd8],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
nine = subprocess.Popen([cmd9],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
ten = subprocess.Popen([cmd10],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
eleven = subprocess.Popen([cmd11],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twelve = subprocess.Popen([cmd12],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
thirteen = subprocess.Popen([cmd13],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
fourteen = subprocess.Popen([cmd14],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
fifteen = subprocess.Popen([cmd15],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
sixteen = subprocess.Popen([cmd16],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
seventeen = subprocess.Popen([cmd17],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
eighteen = subprocess.Popen([cmd18],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
nineteen = subprocess.Popen([cmd19],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty = subprocess.Popen([cmd20],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_one = subprocess.Popen([cmd21],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_two = subprocess.Popen([cmd22],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_three = subprocess.Popen([cmd23],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_four = subprocess.Popen([cmd24],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_five = subprocess.Popen([cmd25],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_six = subprocess.Popen([cmd26],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_seven = subprocess.Popen([cmd27],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_eight = subprocess.Popen([cmd28],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
twenty_nine = subprocess.Popen([cmd29],shell=True,stdin=None, stdout=None, stderr=None)
time.sleep(1)
thirty = subprocess.Popen([cmd30],shell=True,stdin=None, stdout=None, stderr=None)

cmds = [cmd1,cmd2,cmd3,cmd4,cmd5,cmd6,cmd7,cmd8,cmd9,cmd10,cmd11,cmd12,cmd13,cmd14,cmd15,cmd16,cmd17,cmd18,cmd19,cmd20, \
cmd21,cmd22,cmd23,cmd24,cmd25,cmd26,cmd27,cmd28,cmd29,cmd30]

timestamp = datetime.datetime.now()
copyfile("infer.py", "experiments/infer"+logdir+str(timestamp)+".py")
#copyfile("./src/models/finetuning.py", "experiments/finetuning"+logdir+str(timestamp)+".py")
#copyfile("./src/modules/embedding_propagation.py", "experiments/embedding_propagation"+logdir+str(timestamp)+".py")

with open("experiments/"+logdir+str(timestamp)+".txt",'w') as fp:
    for cmd in cmds:
        fp.write(cmd+'\n')
