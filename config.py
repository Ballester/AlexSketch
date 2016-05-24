"""Hyper Parameters"""
epochs = 7
restore_file = 'trained_12179_model.ckpt-12179'	# set to '' if not using

"""Running Options"""
AlexFullSketchTest = True
AlexHalfSketchTest = False
AlexNoSketchTest   = False

AlexFullSketchTrain = False
AlexHalfSketchTrain = False

AlexFromScratchTest = False
AlexFromScratchTrain = False

#Only one can be true
assert AlexFullSketchTest ^ AlexHalfSketchTest ^ AlexNoSketchTest ^ AlexFromScratchTest ^ AlexFullSketchTrain ^ AlexHalfSketchTrain ^ AlexFromScratchTrain

if AlexFullSketchTest:
	save_training = False
	restore_last  = False
	training      = False
	test          = True
	restore_path  = 'full_sketch/'	# with '/'

elif AlexHalfSketchTest:
	save_training = False
	restore_last = False
	training     = True
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
	test = False
	restore_path = 'full_sketch'

elif AlexHalfSketchTrain:
	save_training = True
	restore_last = False
	training = True
	test = False
	restore_path = 'half_sketch'

elif AlexFromScratchTrain:
	save_training = True
	restore_last = False
	training = True
	test = True
	restore_path = 'full_sketch_scratch/'

