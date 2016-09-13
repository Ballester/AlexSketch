import os
for i in range(0, 4):
    for p in range(5, 56, 5):
        os.system('python alexnet.py -l ' + '0.00001' + ' -p ' + str(p)  + ' -i ' + str(i))

