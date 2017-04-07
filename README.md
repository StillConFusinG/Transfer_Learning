# Transfer Learning: How it works!
The idea is to take advantage of deep architectures out there to accomplish quality results on your dataset without taking the pain of training deep networks with lots of data requiring efficient hardwares like fast CPUs/GPUs. Approach is very simple: consider the network comprising of three entities: `Input Layer` -> `hidden units` -> `Output/Logistic layer`. 

`Input Layer` is where we feed our data.

`hidden units` are the layers which transform the input to meaningful abstractions. This entity has millions of parameters which are learned during training of network. This requires lot of data. To give you a feel of "lots of data", take a look at [ImageNet](http://www.image-net.org) dataset. This consists of millions of images spread of 1000 classes. Obviously, for your custom dataset, you can't have that much images and so learning this much parameters with small dataset is practically impossible. However, deep networks, trained on a large datasetare found to generalize to unknown samples and do output abstractions which can be used for classification. So, what we do is simply pass our input through the network to obtain features needed for our classifier.

`Output/Logistic Layer` which serves as our classifier. This is the layer which we essentially train in transfer learning. You can use the obtained features from 2nd entity to tune the already existing Softmax classifier or a custom classifier of your own. Here the Softmax layer of VGG-CNN is discarded and Binary SVM is learned. 

The SVM implementation is borrowed from [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). This is binary SVM and is very fast in nature as it is linear with minimum coding required and the same code generalizes to multi-class scenario. 

Run the above code using these steps. `data_kaggle_catsNdogs.tar.bz2` contains features extracted usoing VGG-CNN in matlab. Code for that also given. The python SVM implemented here also contains [hyper-parameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) which delas with model selection and reducing over-fitting of training data.
```
~: $ git clone https://github.com/vishalkg/Transfer_Learning.git
~: $ tar xvjf data_kaggle_catsNdogs.tar.bz2
~: $ python3 pysvm_catsNdogs.py
```
