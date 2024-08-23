# IMDB Movie/Series Genre Classification

## Project Overview

This project focuses on genre-based text classification using movie and series plot summaries from IMDB. I use a pre-trained DistilBERT model to classify plot summaries into one of four genres: Action, Comedy, Romance, or Horror. The primary goal is to fine-tune the model to achieve high accuracy while minimizing overfitting.

## Dataset Description

The dataset, sourced from Kaggle, contains 1000 IMDB movie/series plot summaries, evenly distributed across four genres: Action, Comedy, Romance, and Horror. Each instance in the dataset includes:

- **ID**: A unique identifier for each movie/series.
- **Plot Summary**: A brief description of the movie/series plot.
- **Genre**: The target label indicating the genre.

The dataset is balanced, with an equal number of examples from each genre. It is split into training (80%) and testing (20%) sets, with the training set further split into training (80%) and validation (20%) subsets.

## Model and Baseline

I employed a DistilBERT pre-trained model as the base model. The initial baseline parameters were set with a learning rate of 5e-5. The baseline model achieved an accuracy of 0.73 on the validation set.

## Evaluation Metrics

The primary metric for model evaluation is accuracy. Additionally, a confusion matrix was used to analyze misclassification patterns, helping to identify common misclassification errors and areas for improvement.

## Misclassification Analysis

After the initial training, I analyzed the misclassified cases:

- **Predicted Comedy when True is Romance**: 14 cases
- **Predicted Romance when True is Comedy**: 8 cases
- **Predicted Comedy when True is Horror**: 7 cases
- **Predicted Horror when True is Comedy**: 7 cases

The confusion matrix provided further insights into these misclassifications, highlighting the challenges in distinguishing between similar genres like Comedy and Romance.

## Overfitting Prevention Strategies

To mitigate overfitting, several strategies were employed:

- **L2 Regularization**: Added to the loss function to penalize large coefficients and improve generalization.
- **Learning Rate Adjustment**: The learning rate was fine-tuned to ensure stable convergence without oscillation.
- **Early Stopping**: Implemented to monitor validation performance and halt training when overfitting is detected, with the patience parameter set to 2 epochs.

## Hyperparameter Tuning

The following hyperparameter configurations were explored:

- **L2 Regularization**: Ranging from 0.01 to 0.06.
- **Learning Rate**: Reduced to 3e-5.
- **Dropout Rate**: Varied between 0.3 and 0.4 to balance regularization.
- **Early Stopping Patience**: Adjusted to find the optimal balance between underfitting and overfitting.

The final model achieved an accuracy of 0.765 with the following parameters:

- **L2 Regularization**: 0.06
- **Learning Rate**: 3e-5
- **Dropout**: 0.37
- **Early Stopping Patience**: 2 epochs

## Final Model Performance

The final model's performance was summarized in the following figures:

- **Training and Validation Loss**: The training loss decreased steadily, while the validation loss showed a slight increase after 2.5 epochs, indicating well-controlled overfitting.
- **Training and Validation Accuracy**: The final validation accuracy was 0.765, indicating the model's strong ability to generalize.

## Conclusion

The final model demonstrated strong performance on the validation set, achieving an accuracy of 0.765. The strategies employed to prevent overfitting, such as L2 regularization, learning rate adjustment, and dropout, were effective in improving the model's generalization capability.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-text-classification).
