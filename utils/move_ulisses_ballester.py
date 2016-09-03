from glob import glob
import shutil
import re
from collections import defaultdict

import os
if os.getcwd().find('utils') == -1:
    raise Exception('You gotta run this from AlexSketch/utils')

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
        synsets.append(line[0].replace(' ', ''))
        synsets_names.append(line[1:])


number_of_images = defaultdict(int)

for name in names:
    remember_name = str(name)
    name = re.split('[/_]', name)[-2]
    try:
        #print name, synsets.index(name)
        sketch_folder_name = sketch_names[sketch_classes.index(synsets.index(name))]
    
        directory_set = '../imagenet_set/' + sketch_folder_name
        if not os.path.exists(directory_set):
            os.makedirs(directory_set)
    
        directory_test = '../imagenet_test/' + sketch_folder_name
        if not os.path.exists(directory_test):
            os.makedirs(directory_test)

        if number_of_images[name] < 60:
            shutil.move(remember_name, directory_set + '/' + str(number_of_images[name]) + '.jpg')
        elif number_of_images[name] < 80:
            shutil.move(remember_name, directory_test + '/' + str(number_of_images[name]) + '.jpg')
        #print name, directory_set + '/' + str(number_of_images[name]) + '.jpg'

        number_of_images[name] += 1
    except:
        print name, synsets_names[synsets.index(name)]
print len(number_of_images)
#print sketch_names[sketch_classes.index(949)]
