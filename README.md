# Cat-Dog Classification using CNN

## Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset used for training and validation is sourced from Kaggle. Additionally, the project includes a functionality to test the model on new images using OpenCV (`cv2`).


## Dataset
- **Source**: [Kaggle - Cat-Dog Dataset](https://www.kaggle.com/)
- The dataset consists of labeled images of cats and dogs used for supervised learning.



## Model Architecture
The CNN model is built with the following architecture:
- **Input Layer**: Image dimensions (resized to 128x128).
- **Convolutional Layers**: Feature extraction using multiple filters.
- **Pooling Layers**: Dimensionality reduction using MaxPooling.
- **Fully Connected Layers**: Dense layers for classification.
- **Output Layer**: Binary classification (Cat: 0, Dog: 1).


## Technologies Used
- **Python**
- **TensorFlow / Keras** for building the CNN model.
- **OpenCV (cv2)** for testing on new images.
- **NumPy**, **Pandas**, and **Matplotlib** for data handling and visualization.



2. **Testing with New Images**:
   - Use the `test_with_cv2.py` script to test the model with new images.
   - Replace `path_to_your_image.jpg` with the path to the image you want to classify.
   - Run:
     ```bash
     python test_with_cv2.py --image path_to_your_image.jpg
     ```

3. **Sample Prediction Output**:
   - The script will print the classification result (Cat or Dog) and display the input image using OpenCV.

---

## Results
- Achieved **accuracy**: *xx%* on the validation set.
- Tested the model on unseen images using OpenCV with promising results.


## Conclusion
This project demonstrates the effectiveness of Convolutional Neural Networks in image classification tasks. By training on a robust dataset and testing on unseen images, the model achieves satisfactory performance in distinguishing between cats and dogs. The inclusion of OpenCV functionality enhances the project’s usability, making it easier to evaluate the model’s predictions in real-world scenarios. Future improvements could involve fine-tuning the model or exploring transfer learning techniques to achieve higher accuracy.



### Acknowledgments
- Kaggle for the Cat-Dog dataset.
- TensorFlow and OpenCV communities for extensive documentation and support.

