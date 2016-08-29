import os
import urllib

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


for idx, sketch in enumerate(sketch_classes):
    print synsets[sketch], synsets_names[sketch]
    directory_set = '../imagenet_set/' + sketch_names[idx]
    if not os.path.exists(directory_set):
        os.makedirs(directory_set)
    directory_test = '../imagenet_test/' + sketch_names[idx]
    if not os.path.exists(directory_test):
        os.makedirs(directory_test)

    links = urllib.urlopen('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + synsets[sketch]).read()
    for link in links:
        
    break

        

