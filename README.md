# IMDB Movie/Series Genre Classification

## Project Overview

This project focuses on genre-based text classification using movie and series plot summaries from IMDB. The project fine-tunes a pre-trained DistilBERT transformer model to classify plot summaries into one of four genres: Action, Comedy, Romance, or Horror. The model is built using TensorFlow and the Hugging Face Transformers library, fine-tuning the pre-trained DistilBERT base model rather than training from scratch.

## Dataset Description

The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-text-classification), contains 1000 IMDB movie/series plot summaries, evenly distributed across four genres: Action, Comedy, Romance, and Horror. Each instance in the dataset includes:

- **ID**: A unique identifier for each movie/series
- **Description**: A brief plot summary of the movie/series
- **Genre**: The target label indicating the genre (action, comedy, horror, or romance)

The dataset is balanced, with 250 examples from each genre. The data is split into:
- **Training set**: 640 samples (80% of total)
- **Validation set**: 160 samples (20% of training data)
- **Test set**: 200 samples (20% of total)

## Model Architecture

The project uses **DistilBERT** (distilbert-base-uncased), a lightweight, distilled version of BERT that maintains most of BERT's performance while being faster and smaller. The model is fine-tuned for multi-class classification with 4 output labels.

### Baseline Model Configuration

The baseline model uses the following configuration:

- **Pre-trained Model**: DistilBERT base uncased
- **Optimizer**: Adam (legacy)
- **Learning Rate**: 5e-5
- **Loss Function**: Categorical Crossentropy (from_logits=True)
- **Regularization**: L2 regularization with lambda=0.01
- **Batch Size**: 8 for training, 16 for validation and testing
- **Epochs**: 10
- **Metrics**: Accuracy

## Training Process

The model was trained for 10 epochs with the following results:

- **Final Training Accuracy**: 100%
- **Final Validation Accuracy**: 72.92%
- **Test Accuracy**: 73%

The training process revealed significant overfitting, with training loss decreasing to near zero while validation loss increased after the initial epochs. This indicates the model was memorizing training patterns rather than learning generalizable features.

## Evaluation and Analysis

### Test Set Performance

The baseline model achieved **73% accuracy** on the unseen test set, with a test loss of 1.28.

### Misclassification Analysis

Analysis of misclassified cases revealed the following patterns:

- **Predicted Comedy when True is Romance**: 14 cases
- **Predicted Romance when True is Comedy**: 8 cases
- **Predicted Comedy when True is Horror**: 7 cases
- **Predicted Horror when True is Comedy**: 7 cases

The confusion matrix shows that the model struggles most with distinguishing between Comedy and Romance genres, which often share similar narrative elements and emotional tones.

### Overfitting Indicators

The learning curves clearly demonstrate overfitting:

- Training loss rapidly decreases to near zero
- Validation loss initially decreases but then increases steadily
- Training accuracy reaches 100% while validation accuracy plateaus around 72-73%
- Large gap between training and validation performance indicates poor generalization

## Next Steps for Improvement

The baseline model identified several strategies to improve generalization:

1. **L2 Regularization**: Adjust the regularization strength to better control model complexity
2. **Learning Rate Tuning**: Experiment with lower learning rates (e.g., 3e-5) for more stable convergence
3. **Early Stopping**: Implement early stopping to halt training when validation performance degrades
4. **Dropout**: Consider adding dropout layers to prevent overfitting

## Usage

The notebook includes a prediction function that allows testing on custom text:

```python
prediction(text)
```

This function takes a movie plot summary as input and returns the predicted genre along with probability scores for all four genres.

## Project Structure

- `BaseLine_Model.ipynb`: Baseline model implementation with full training and evaluation pipeline
- `AdjustingParameters.ipynb`: Hyperparameter tuning experiments (if available)
- `IMDB_larger_description_dataset.csv`: The dataset used for training
- `logs/`: TensorBoard log files for training and validation metrics

## Dependencies

Key libraries used:
- TensorFlow
- Transformers (Hugging Face)
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

See `requirements.txt` for complete dependency list.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-text-classification)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
