# Digit Recognition Using Convolutional Neural Networks (CNN)

This project involves training a Convolutional Neural Network (CNN) to recognize hand-drawn digits from a dataset of grayscale 28x28 pixel images. The model is trained using the `train.csv` dataset, which contains labeled images of digits (0-9) drawn by users.

## Dataset Description

The dataset consists of the following CSV files:

- **train.csv**: Contains 785 columns. The first column is the "label", representing the digit drawn by the user. The remaining 784 columns contain pixel values for a 28x28 grayscale image, with each pixel's intensity ranging from 0 (white) to 255 (black).
- **test.csv**: Contains the pixel values of test images without labels. This file is used to evaluate the model after training.

Each image is represented as a flattened 1D array of 784 pixels (28x28). The pixel columns are named `pixel0`, `pixel1`, ..., `pixel783`, and can be reshaped into a 28x28 matrix.

## Model Architecture

The CNN model consists of several layers designed for digit classification:

- **Convolutional Layers**: Two convolutional layers with 32 and 64 filters respectively, each followed by a ReLU activation function.
- **Batch Normalization**: Applied after each convolutional layer to normalize the inputs.
- **MaxPooling**: After some convolutional layers, max-pooling layers with a 2x2 filter size are used to reduce spatial dimensions.
- **Dropout**: Regularization is applied with dropout layers to prevent overfitting.
- **Fully Connected Layers**: Flattened output from the convolutional layers, followed by a fully connected dense layer with 128 units and ReLU activation.
- **Output Layer**: A softmax activation is applied to output the probabilities of each digit (0-9).

## Training and Evaluation

- **Data Augmentation**: The `ImageDataGenerator` class is used to apply random transformations to the training images, such as rotations, shifts, and zooms, in order to improve model generalization.
- **Learning Rate Scheduling**: Learning rate is adjusted dynamically using `ReduceLROnPlateau` to reduce it if the validation loss plateaus.
- **Early Stopping**: Training stops early if the validation loss doesn't improve for 5 consecutive epochs.
- **Loss Function**: Categorical cross-entropy loss is used, as the problem is a multi-class classification task.
- **Optimizer**: Adam optimizer is used for efficient training.

## Training Results

After training, the model's accuracy and loss are plotted over epochs for both training and validation datasets to visualize its performance.

### Model Accuracy

- **Training Accuracy**: The model achieved a training accuracy of **0.97** on the training data.
- **Validation Accuracy**: The model achieved a validation accuracy of **0.99** on the validation data.

### Model Accuracy Plot
![Model Accuracy](accuracy_plot.png)

### Model Loss Plot
![Model Loss](loss_plot.png)

## Conclusion

This model demonstrates the power of Convolutional Neural Networks (CNNs) in image recognition tasks. By using data augmentation, dropout, and early stopping techniques, the model generalizes well to unseen data and is capable of recognizing digits with high accuracy.

The model can be further improved by experimenting with deeper architectures, more data augmentation techniques, and hyperparameter tuning. However, this implementation provides a solid foundation for digit recognition tasks and can be extended to similar image classification problems.

## Files Included

- `train.csv`: Training dataset with images and labels.
- `test.csv`: Test dataset without labels.
- `digit_recognition_model.py`: The Python script containing the CNN model and training code.
- `accuracy_plot.png`: The plot of training and validation accuracy.
- `loss_plot.png`: The plot of training and validation loss.
