import os
import urllib
import urllib2
import socket
socket.setdefaulttimeout(10)


class HeadRequest(urllib2.Request):
    def get_method(self):
        return "HEAD"

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


    counter = 0
    links = urllib2.urlopen('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + synsets[sketch]).read()
    for link in links.split('\n'):
        
        try:
            response = urllib2.urlopen(HeadRequest(link), timeout=5)
            if response.info()["Content-type"].find('image') != -1:
                print link
                if counter < 60:
                    try:
                        urllib.urlretrieve(link, os.path.join(directory_set, str(counter) + '.' + link.split('.')[-1]))
                        counter += 1
                    except:
                        raise Exception('Download failed')
                elif counter < 80:
                    try:
                        urllib.urlretrieve(link, os.path.join(directory_test, str(counter) + '.' + link.split('.')[-1]))
                        counter += 1
                    except:
                        raise Exception('Download failed')
                else:
                    break
        except:
            pass



        

