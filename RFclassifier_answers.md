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

- Overall accuracy: 92.67%

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
All channels accuracy: 0.9267
RGB-only accuracy: 0.8933
```

This was within expectations because NIR contains information that is not visible to the human eye but is important for land cover classification. For example, vegetation reflects strongly in NIR. Without this channel, some classes become harder to distinguish, leading to the accuracy drop.

#### 2.4: Make use of only R, G, and NIR. What changes?

An increase of classification accuracy was observed:

```
R + G + NIR accuracy: 0.9183
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

