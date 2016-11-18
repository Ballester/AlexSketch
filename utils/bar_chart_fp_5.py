from collections import defaultdict
from collections import Counter
from operator import add

import matplotlib.pyplot as plt
from caffe_classes import class_names

fp5s = []
for i in range(1000, 1001):
    with open('results/PA_57--LR_0.001--E_1--' + str(i) + '.txt') as fid:
        for line in fid:
            if line.find('False Positives 5') != -1:
                fp5s.append(eval(line.split(':', 1)[1]))



#fp5 = sum([Counter(x) for x in fp5s]) / float(len(fp5s))
dict_sum = {}
for x in fp5s:
    dict_sum = dict(Counter(x)+Counter(dict_sum))

#print dir(dict_sum)
for k, v in dict_sum.items():
    dict_sum[k] = v/1.0

#print dict_sum

names = []
#print len(class_names)
with open('utils/one-hot_references.txt') as fid:
    fid.readline()
    for line in fid:
        names.append(class_names[int(line.split(':')[1])])

dict_aux = {}
for name in names:
    try:
        dict_aux[name] = dict_sum[name]
    except:
        dict_aux[name] = 0

#print len(dict_sum)
#dict_sum = dict(Counter(dict_sum)+Counter(dict_aux))

print len(dict_aux)
plt.bar(range(0, 57), dict_aux.values())
plt.savefig('bar_chart_fp5_0.pdf')

#print Counter(fp5s[0])+Counter(fp5s[1])/2
#a = map(add, fp5s[0], fp5s[1])
#print a
