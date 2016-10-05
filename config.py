from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--learning_rate", dest="learning_rate",
                  help="learning rate value", default=0.001)
parser.add_option("-e", "--epochs", dest="epochs",
                  help="number of epochs to train", default=1)
parser.add_option("-p", "--partial_amount", dest="partial_amount",
                  help="partial amount of half sketch training", default=57)
parser.add_option("-r", "--restore_file", dest="restore_file",
                  help="which file to restore", default='')
parser.add_option("-i", "--iteration", dest="iteration",
                  help="iteration at the moment", default="1000")

(options, args) = parser.parse_args()

"""Hyper Parameters"""
epochs = int(options.epochs)
partial_amount = int(options.partial_amount)
restore_file = options.restore_file   # set to '' if not using
learning_rate = float(options.learning_rate)
iteration = int(options.iteration)

"""Running Options"""
AlexFullSketchTest = False
AlexHalfSketchTest = False
AlexNoSketchTest   = True

AlexFullSketchTrain = False
AlexHalfSketchTrain = False

AlexFromScratchTest = False
AlexFromScratchTrain = False

#Only one can be true
assert AlexFullSketchTest ^ AlexHalfSketchTest ^ AlexNoSketchTest ^ AlexFromScratchTest ^ AlexFullSketchTrain ^ AlexHalfSketchTrain ^ AlexFromScratchTrain

if AlexFullSketchTest:
    save_training = False
    restore_last  = True
    training      = False
    test          = True
    restore_path  = 'full_sketch/'  # with '/'

elif AlexHalfSketchTest:
    save_training = False
    restore_last = True
    training     = False
    test         = True
    restore_path = 'half_sketch/'

elif AlexNoSketchTest:
    save_training = False
    restore_last = False
    training = False
    test = True
    restore_path = 'no_sketch/'

elif AlexFromScratchTest:
    save_training = False
    restore_last = True
    training = False
    test = True
    restore_path = 'full_sketch_scratch/'

elif AlexFullSketchTrain:
    save_training = True
    restore_last = False
    training = True
    test = True
    restore_path = 'full_sketch/'

elif AlexHalfSketchTrain:
    save_training = True
    restore_last = False
    training = True
    test = True
    restore_path = 'half_sketch/'

elif AlexFromScratchTrain:
    save_training = True
    restore_last = False
    training = True
    test = True
    restore_path = 'full_sketch_scratch/'

