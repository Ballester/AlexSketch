import os
import numpy as np
from natsort import natsorted
from collections import defaultdict

files = os.listdir('results/')
results = defaultdict(list)

for f in natsorted(files):
    f = f.rsplit('--', 1)

    with open('results/' + f[0] + '--' + f[1].replace('.txt', "") + '.txt') as fid:
        preliminar = {}
        for line in fid:
            line = line.split(': ')
            if line[0] == 'Top-5':
                preliminar[line[0]] = int(line[1])
            elif line[0] == 'Top-1':
                preliminar[line[0]] = int(line[1])
            elif line[0] == 'Trained Top-5':
                preliminar[line[0]] = int(line[1])
            elif line[0] == 'Trained Top-1':
                preliminar[line[0]] = int(line[1])
            elif line[0] == 'Not-Trained Top-5':
                preliminar[line[0]] = int(line[1])
            elif line[0] == 'Not-Trained Top-1':
                preliminar[line[0]] = int(line[1])
        results[f[0]].append(preliminar)
        #print results[f[0]]
        #print f[0], preliminar

for k, v in results.iteritems():
    print 'Files: ', k
    top_5 = []
    top_1 = []
    trained_top_5 = []
    trained_top_1 = []
    not_trained_top_5 = []
    not_trained_top_1 = []
    for res in v:
        top_5.append(res['Top-5'])
        top_1.append(res['Top-1'])
        trained_top_5.append(res['Trained Top-5'])
        trained_top_1.append(res['Trained Top-1'])
        not_trained_top_5.append(res['Not-Trained Top-5'])
        not_trained_top_1.append(res['Not-Trained Top-1'])

    print 'Mean Top-5: ', np.mean(top_5), ' Std Dev: ', np.std(top_5)
    print 'Mean Top-1: ', np.mean(top_1), ' Std Dev: ', np.std(top_1)
    print 'Mean Trained Top-5: ', np.mean(trained_top_5), ' Std Dev: ', np.std(trained_top_5)
    print 'Mean Trained Top-1: ', np.mean(trained_top_1), ' Std Dev: ', np.std(trained_top_1)
    print 'Mean Not-Trained Top-5: ', np.mean(not_trained_top_5), ' Std Dev: ', np.std(not_trained_top_5)
    print 'Mean Not-Trained Top-1: ', np.mean(not_trained_top_1), ' Std Dev: ', np.std(not_trained_top_1) 
