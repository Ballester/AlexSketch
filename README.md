# Lateral Representation Learning in Convolutional Neural Networks
### As seen in IJCNN 2018 proceedings (to be released)

This repository implements some of the experiments presented on the paper. 

**Abstract:** _We explore a type of transfer learning in Convolutional Neural Networks where a network trained on a primary representation of examples (e.g. photographs) is capable of generalizing to a secondary representation (e.g. sketches) without fully training on the latter. We show that the network is able to improve classification on classes for which no examples in the secondary representation were provided, an evidence that the model is exploiting and generalizing concepts learned from examples in the primary representation. We measure this lateral representation learning on a CNN trained on the ImageNet dataset and use overlapping classes in the TU-Berlin and Caltech-256 datasets as secondary representations, showing that the effect can't be fully explained by the network learning newly specialized kernels. This phenomenon can potentially be used to train classes in domain adaptation tasks for which few examples in a target representation are available._


Prerequisites:

  * Tensorflow

Expects a file with the intersecting classes between source and target datasets.
One way to find this is using https://github.com/Ballester/Dataset-Class-Relationship/
