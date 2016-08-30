import os
for i in range(0, 30):
    for p in range(5, 56, 5):
        for l in range(1, 6, 1):
            l = 10**(-l)
            os.system('python alexnet.py -l ' + str(l) + ' -p ' + str(p) + ' -i ' + str(i))

