# Tank Image Classification

Multiclass image classification of tank models from game footage using deep learning.

## Task
Classify tank images into predefined classes based on visual appearance.

## Model
- EfficientNet-B0
- Transfer Learning from ImageNet
- PyTorch

## Dataset
- ~500 images per class
- Train / Validation / Test split

## Results
'''
              precision    recall  f1-score   support

           0      0.980     0.980     0.980        50
           1      0.960     0.960     0.960        50
           2      0.980     0.960     0.970        50
           3      1.000     0.920     0.958        50
           4      0.907     0.980     0.942        50
           5      1.000     0.980     0.990        50
           6      0.980     0.960     0.970        50
           7      0.979     0.940     0.959        50
           8      0.980     1.000     0.990        50
           9      0.926     1.000     0.962        50

    accuracy                          0.968       500
   macro avg      0.969     0.968     0.968       500
weighted avg      0.969     0.968     0.968       500
'''
