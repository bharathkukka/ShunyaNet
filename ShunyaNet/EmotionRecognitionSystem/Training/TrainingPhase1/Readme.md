# Emotion Recognition - Phase 1 Training Report

## Introduction
This is my summary and reflection on Phase 1 of training my emotion recognition model. The goal was to get a baseline on my dataset and see how well the model could distinguish between different emotions using a straightforward setup.

## Training Setup
- **Epochs:** 52
- **Early Stopping Patience:** 35
- **Batch Size:** 16
- **Hardware:** Intel i7 12th Gen CPU

## Results
### Test Performance
- **Test Loss:** 2.0376
- **Test Accuracy:** 0.2091

### Classification Report
```
              precision    recall  f1-score   support

       anger     0.0000    0.0000    0.0000       323
    contempt     0.0000    0.0000    0.0000       288
     disgust     0.0000    0.0000    0.0000       249
        fear     0.0000    0.0000    0.0000       319
       happy     0.4444    0.0317    0.0591       505
     neutral     0.2779    0.6732    0.3934       514
         sad     0.0000    0.0000    0.0000       310
    surprise     0.1513    0.6099    0.2425       405

    accuracy                         0.2091      2913
   macro avg     0.1092    0.1643    0.0869      2913
weighted avg     0.1471    0.2091    0.1134      2913
```

### Training History
- Early stopping triggered at epoch 35.
- Training and validation accuracy hovered around 0.14–0.17 for most epochs, with a slight increase towards the end.
- Training and validation loss started high and decreased slowly, but never reached low values. There were some spikes (e.g., epoch 4 and 6) where loss values were abnormally high, possibly due to instability or data issues.
- The best validation accuracy was about 0.18, but the final test accuracy was only 0.21.

### Visualizations


**Confusion Matrix**  
![Confusion Matrix](ShunyaNet/EmotionRecognitionSystem/Training/TrainingPhase1/results/test_confusion_matrix.png)


**Training History**  
![Training History](ShunyaNet/EmotionRecognitionSystem/Training/TrainingPhase1/results/training_history.png)

## Observations
This phase was a bit disappointing in terms of raw numbers. The model only managed about 21% accuracy on the test set, which is not much better than random guessing (given 8 classes). Most classes like 'anger', 'contempt', 'disgust', 'fear', and 'sad' had zero precision and recall, meaning the model never predicted them correctly. Only 'neutral' and 'surprise' had reasonable recall, but even there, precision was low.

Looking at the training curves, I noticed some instability in the loss values, especially early on. There might be issues with the data distribution or class imbalance, since the model seems to default to predicting 'neutral' and 'surprise' most of the time. Early stopping kicked in at epoch 35, but it didn't really help the model generalize better.

Overall, this phase highlighted the challenges of emotion recognition with this dataset. The model is not yet able to distinguish between most emotions, and I suspect class imbalance and possibly data quality are major factors. For the next phase, I plan to address these issues, maybe by augmenting the data, rebalancing the classes, or trying a different architecture.

---