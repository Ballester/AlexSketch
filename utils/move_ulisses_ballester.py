from glob import glob
import shutil
import re

names = glob('/home/ulisses/SubImageNet/*.JPEG')

"""Open reference file"""
sketch_classes = []
sketch_names = []
with open('one-hot_references.txt') as fid:
    # dump
    fid.readline()
    one_hots = []
    one_hots_values = []
    for i in range(0, 57):
        aux = fid.readline()
        aux = aux.split(':')
        sketch_classes.append(int(aux[1]))
        sketch_names.append(aux[0])

synsets = []
synsets_names = []
with open('synset_words.txt') as fid:
    for line in fid:
        line = line.split(' ')
        synsets.append(line[0])
        synsets_names.append(line[1:])

for name in names:
    name = re.split('[/_]', name)
    print name[-2], sketch_names[sketch_classes.index(synsets.index(name[-2]))]
    #print sketch_names[sketch_classes.index(949)]
    break
