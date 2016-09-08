import os
for i in range(0, 30):
    os.system('python alexnet.py -l ' + '0.00001'  + ' -i ' + str(i))

