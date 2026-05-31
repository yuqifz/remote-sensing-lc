# HW1: Remote sensing for landcover detection: ML techniques



## Overview

The goal of this homework is to classify remote sensing images into six land-cover categories using machine learning methods. Two approaches were investigated:

1. Random Forest classifier
2. Convolutional Neural Network (CNN)


For both models, I evaluated classification performance, analyzed per-class accuracy, visualized correct and incorrect predictions, and investigated the influence of different input channels and model hyperparameters.

## Dataset

The data for these demonstrations is a subset from the [SAT-6](https://csc.lsu.edu/~saikat/deepsat/) dataset by *Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015.* It consists of 28x28 pixels uint8 images with 4 channels and labels of 6 landcover classes - barren land, trees, grassland, roads, buildings and water bodies.

The classification task is scene classification, i.e. the entire 28x28 pixel image is classified a sone landcover type.



## Random Forest classifier

#### 2.1 Notebook questions

##### *Question:* What would you need to do to extract only the green and the infrared channel from this data?

Order of columns:

R G B NIR | R G B NIR | R G B NIR …

To extract the G and NIR channels:

```python
# slicing: start:stop:step
green = landcover_df[:,1::4] 
infr = landcover_df[:,3::4]
```



##### *Question:* What is the advantage of this encoding compared to a simple class label like '0', '1', '2', '3', '4', '5', or text labels like 'building', 'barren_land', ...?

Numerical labels falsely imply order and numerical relationships between classes, while text labels are strings which cannot be used directly for mathematical operations. One-hot encoding solves both issues.

##### *Question:* Why use `extend`here and `append` above?

`np.random.choice(...).tolist()` returns a list of indices. `extend` unpacks the array and adds each element individually, resulting in a flat list.

`append` adds each index list as a nested list, resulting in a nested list.

```python
a = []
a.append([1,2,3])
a.append([4,5,6])
# [[1,2,3], [4,5,6]]

b = []
b.extend([1,2,3])
b.extend([4,5,6])
# [1,2,3,4,5,6]
```



##### *Question:* What is wrong with the above code?

The code samples `train_idx` and `test_idx` from the same pool of class indices. This can lead to samples appearing in both the training and test sets, causing data leakage. Test samples should be selected from the remaining pool after training samples are removed.



##### _Question:_ Why do you want to shuffle the samples in the train and test datasets?

Without doing so, the data is ordered by class. Shuffling ensures the samples of different classes are randomly distributed. This helps prevent the model from learning unintended patterns caused by ordering of the data.



#### 2.2: Add some code to the notebook to evaluate the classification accuracy of each class and to plot a small number of correct classification examples and a small number of misclassifications.

- Overall accuracy: 91.67%

- Per-class results:

  ```
  								precision   recall  f1-score    support
  
      building       0.87      0.97      0.92       100
   barren_land       0.98      0.92      0.95       100
         trees       0.92      0.98      0.95       100
     grassland       0.90      0.82      0.86       100
          road       0.97      0.93      0.95       100
         water       1.00      1.00      1.00       100
  
      accuracy                           0.94       600
     macro avg       0.94      0.94      0.94       600
  weighted avg       0.94      0.94      0.94       600
  ```



#### 2.3: Inspect the 4 color channels of the data (R, G, B, and NIR). Modify the code so that it only uses the R, G, and B channels. How does this affect the classification accuracy?

After removing the NIR channel, the classification accuracy dropped:

```
All channels accuracy: 0.9167
RGB-only accuracy: 0.8833
```

This was within expectations because NIR contains information that is not visible to the human eye but is important for land cover classification. For example, vegetation reflects strongly in NIR. Without this channel, some classes become harder to distinguish, leading to the accuracy drop.

#### 2.4: Make use of only R, G, and NIR. What changes?

An increase of classification accuracy was observed:

```
R + G + NIR accuracy: 0.9000
```

This suggests that NIR is more informative than the blue channel for this task.

#### 2.5: Look at the documentation of random forest and its hyper parameters. Choose one hyperparameter and modify its value. Justify your choice of the hyper parameter and of the new value. What is your expectation? Will the classification accuracy increase or decrease? Will the model become more robust or less robust? Implement the change and document your results. Do the results agree with your expectation?

**Choice:** `n_estimators` (number of trees), increased from 100 to 500. **Justification:** More trees mean the model averages over a larger number of decision trees.

**Expectation:** Accuracy should slightly increase or remain similar;  the model should become more robust.

**Results:** 

```
Baseline n_estimators=100 accuracy: 0.9250
Modified n_estimators=500 accuracy: 0.9217
```

**Discussion:** The small decrease in accuracy is likely due to random variation inherent in the training process rather than a performance drop. With more trees, the model is theoretically more stable, but the marginal gain diminishes quickly after a certain threshold. The result is broadly consistent with the expectation.



## CNN Classifier

A fixed random seed was used to improve the reproducibility of the experiments.

#### 3.2: Add some code to the notebook to evaluate the classification accuracy of each class and to plot a small number of correct classification examples and a small number of misclassifications. Document what you need to change from task 2.2.

```
                precision    recall  f1-score   support

    building       0.94      0.88      0.91       100
 barren_land       0.96      0.95      0.95       100
       trees       1.00      0.91      0.95       100
   grassland       0.87      0.96      0.91       100
        road       0.89      0.94      0.91       100
       water       1.00      1.00      1.00       100

    accuracy                           0.94       600
   macro avg       0.94      0.94      0.94       600
weighted avg       0.94      0.94      0.94       600
```

##### Changes from 2.2:

1. **Getting predictions:** In 2.2, predictions were obtained by calling `rf.predict(test_X)`  and converting one-hot outputs with `np.argmax`. Here, `pred_labels` and  `true_labels` are collected during the training loop as tensors, so they only need to be converted to numpy with `.item()`. 
2. **Class name variable:** The variable `labels` used in 2.2 was overwritten during the CNN training loop (`data, labels = train_data`). Therefore, class names must be stored in a separate variable `class_names = annotations[0].values` (reading them again).



#### 3.3: Modify the CNN classifier so that it only uses the R, G, and B channels. How do the results degrade? Is the effect smaller or larger than with the random forest?

```
All channels accuracy: 94.0% 
RGB-only accuracy: 93.0% 
```

Compared to the random forest experiment(2.3), the effect of removing NIR is smaller(the accuracy dropped by ~1%). The CNN can learn spatial features directly from RGB channels through its convolutional layers, partially compensating for the missing NIR information.



#### 3.4 Choose one hyperparameter of the CNN architecture or the training and modify its value. Justify your choice of the hyper parameter and of the new value. What is your expectation? Will the classification accuracy increase or decrease? Will the model become more robust or less robust? Implement the change and document your results. Do the results agree with your expectation?

**Choice:** learning rate, reduced from 0.001 to 0.0001. 

**Justification:** The learning rate controls how large each parameter update  step is during training. With only 10 epochs, a smaller learning rate means  the model takes smaller steps and may not converge fully within the same  number of epochs. 

**Expectation:** Accuracy will likely decrease with only 10 epochs, as the model needs more epochs to reach the same convergence point. 

**Results:** 

```
Original (lr=0.001):  94.0%
New (lr=0.0001):      84.0%
```

**Discussion:** The results agree with the expectation. The significant drop of  ~10% confirms that with only 10 epochs, a learning rate of 0.0001 is too small  for the model to converge sufficiently. However, one can always increase the number of epochs when necessary.



## Comparison: Random Forest vs CNN

### Accuracy

| **Model**                    | **Accuracy** |
| ---------------------------- | ------------ |
| Random Forest (all channels) | 0.9167       |
| Random Forest (RGB only)     | 0.8883       |
| CNN (all channels)           | 0.9400       |
| CNN (RGB only)               | 0.9300       |

The CNN outperforms the random forest in both settings, demonstrating the 
advantage of learning spatial features through convolutional layers rather 
than treating each pixel as an independent feature. 

### Training Time

- **Random Forest:** trains in seconds with a single `rf.fit()` call 
- **CNN:** requires 10 epochs of iterative training, significantly slower 

### Inference Time

- **Random Forest:** near-instant prediction with `rf.predict()`
- **CNN:** slightly slower due to forward pass through multiple layers, but still fast for this dataset size



## Potential Improvements for CNN

I would consider modifying the following parameters: 

- **num_epochs:** increasing from 10 to 50+ would allow the model more time to converge
- **learning rate scheduler:** uncommenting the scheduler would gradually reduce the learning rate during training
- **num_train:** increasing training samples per class beyond 1000 would reduce overfitting and improve generalization



## Author

Yuqi Fang

This work is based on the original notebooks by Prof. Dr. Martin Schultz, available at the course repository.

AI tool was used for clarifying concepts, suggesting code structure, and debugging during the development process.
