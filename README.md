# caffe_landmark

A brief implementation for the paper [Deep Convolutional Network Cascade for Facial Point Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf).
Here I only chose the first stage of the architecture mentioned in the paper and used two different datasets to implement respectively, which 
was proved that data is more effective than a sophiscated network.

### Data Processing

**Get data**

The [original data](http://mmlab.ie.cuhk.edu.hk/archive/CNN/) in the paper and the [celeba dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for the second choice.

**Convert to h5**

```
python generate_h5.py
```
or
```
python generate_h5.py celeba
```

### Training

Run the script `train.sh`, you may need to choose your gpu and redirect the log.txt

### Evaluate

Evaluate the mean error for the fiva keypoints.

```
python value_error.py ../model/model_001.caffemodel
```
or
```
python value_error.py ../model/model_002.caffemodel celeba
```
Here I got my final evaluation.

| Mean Error | Original Dataset | Celeba |
| :-----: | :-----: | :------: |
| Validation Images | 3466 | 19867 |
| Lefteye error | 0.026229 | 0.016251 |
| Righteye error | 0.027118 | 0.018063 |
| Nose error | 0.036313 | 0.028944 |
| Leftmouth error | 0.034152 | 0.028052 |
| Rightmouth error | 0.034472 | 0.029255 |

### Predict

```
python predict.py ../model/model_001.caffemodel
```
or
```
python predict.py ../model/model_002.caffemodel celeba
```
![](http://7xvn1q.com2.z0.glb.qiniucdn.com/16-7-27/18084155.jpg)

