import os
for i in range(0, 30):
    for p in range(1, 39):
        os.system('python alexnet.py -e 1 -l ' + '0.00001' + ' -p ' + str(p)  + ' -i ' + str(i))
