import os

for r in range(5, 6, 5):
    os.system("python alexnet.py -r " + "PA_" + str(r) + "--LR_1e-05_1000_model.ckpt-" + str(r*60) + " -p " + str(r))
