# Transfer Learning: How it works!
The idea is to take advantage of deep architectures out there to accomplish quality results on your dataset without taking the pain of training deep networks with lots of data requiring efficient hardwares like fast CPUs/GPUs. Approach is very simple: consider the network comprising of three entities: `Input Layer` -> `hidden units` -> `Output/Logistic layer`. 

`Input Layer` is where we feed our data.

`hidden units` are the layers which transform the input to meaningful abstractions. This entity has millions of parameters which are learned during training of network. This requires lot of data. To give you a feel of "lots of data", take a look at [ImageNet](http://www.image-net.org) dataset. This consists of millions of images spread of 1000 classes. Obviously, for your custom dataset, you can't have that much images and so learning this much parameters with small dataset is practically impossible. However, deep networks, trained on a large dataset are found to generalize to unknown datasets and do output abstractions which can be used for classification. So, what we do is simply pass our input through the network to obtain features needed for our classifier.

`Output/Logistic Layer` which serves as our classifier. This is the layer which we essentially train in transfer learning. You can use the obtained features from 2nd entity to tune the already existing Softmax classifier or a custom classifier of your own. Here the Softmax layer of VGG-CNN is discarded and Binary SVM is learned. 

The SVM implementation is borrowed from [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). This is binary SVM with [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) and is very fast in nature as it is linear with minimum coding required and the same code generalizes to multi-class scenario. More details about using RBF kernel in SVM can be found [here](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html).

Run the below code using these steps. The python SVM implemented here also contains [hyper-parameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) which deals with model selection and reducing over-fitting of training data.

Download [VGG-CNN from here](http://www.vlfeat.org/matconvnet/pretrained/). In order to extract features in batches, you will need `cnnPredict.m`. This is available [here](https://github.com/parallel-forall/code-samples/blob/master/MATLAB_deeplearning/cnnPredict.m). Use `extract_feat_all.m` to extarct features. Use pysvm script to train your SVM classifier for any dataset. Sample extracted features are included in `data_kaggle_catsNdogs.tar.bz2`.

```
~: $ git clone https://github.com/vishalkg/Transfer_Learning.git
~: $ tar xvjf data_kaggle_catsNdogs.tar.bz2
~: $ python3 pysvm_catsNdogs.py
```

A total of 5000 samples were used for tarining and 12250 for testing the performance of classifier. Accuracy obtained after hp-optimization: ~71%. You can use the matlab script to extract features for more samples if you wish to and add it to existing training set to improve further.

Reference: https://devblogs.nvidia.com/parallelforall/deep-learning-for-computer-vision-with-matlab-and-cudnn/
