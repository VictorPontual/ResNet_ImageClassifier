# Plant Seedlings Classification Using ResNet

## Overview

This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** using a **ResNet architecture** to classify plant seedlings into different species. The model leverages **residual connections** for effective training of deep networks. The entire workflow—from data preprocessing, model definition, and training to evaluation—is implemented in a **single Google Colab notebook**.

The model uses the **Plant Seedlings Classification dataset** and is built with **TensorFlow** and the **Keras API**.

## Key Features

- **ResNet Architecture**: Utilizes residual connections to ease the training of deep neural networks.
- **Data Augmentation**: Applies various transformations (rotation, zoom, flip) to augment the dataset and improve model generalization.
- **Callbacks**: Includes **Early Stopping**, **Model Checkpointing**, and **Learning Rate Scheduling** for stable and efficient training.
- **Customizable**: The model and its parameters are flexible, allowing easy adjustments to the number of classes, block configurations, and image input size.

## Project Structure

Since the entire project is implemented in a single Google Colab notebook, there are no separate files or directories. You can access the notebook [here](https://colab.research.google.com/drive/1v2HuXJejg4skRCs1D4z72gR7kQJCJw5C#scrollTo=qO1HbFIJrVZh) (provide a link to your notebook if it's shared publicly).

---

## Setup and Installation

To use this notebook, follow these steps:

1. **Clone the repository** (if applicable) or simply open the Google Colab notebook linked above.
2. **Install dependencies**: The notebook installs the required dependencies within the environment. Run the following cell at the start of the notebook to install the necessary packages:
   ```python
   !pip install tensorflow keras matplotlib sklearn numpy
   ```

3. **Download the Dataset**: The project uses the **Plant Seedlings Classification** dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification). You can upload it directly to your Google Colab environment or use the Kaggle API to download the dataset.

4. **Upload your dataset**: Once downloaded, upload the dataset to the notebook environment. Place the dataset in the appropriate folder (e.g., `Nonsegmented/`).

---

## Workflow in the Notebook

### 1. Data Preprocessing and Augmentation

The notebook first loads and preprocesses the **Plant Seedlings Classification** dataset. To improve model generalization, **data augmentation** is applied. Transformations include:
- Rotation
- Zoom
- Horizontal and Vertical Flip
- Width and Height Shifts
- Shearing

### 2. Model Architecture

The core of the model is a **ResNet-based architecture**, which includes several **residual blocks** to make the training of deep networks more efficient. Each block allows the model to learn both the identity mapping and more complex transformations.

The architecture includes:
- **Convolutional Layers**: To extract features from input images.
- **Residual Blocks**: Using skip connections to allow gradients to flow easily during backpropagation.
- **Global Average Pooling**: To reduce the spatial dimensions before classification.
- **Fully Connected Layer**: To classify images into the different species.

### 3. Callbacks for Training

The following **callbacks** are used to enhance the training process:
- **ModelCheckpoint**: Saves the model with the best validation accuracy.
- **EarlyStopping**: Stops training early if the validation loss does not improve for a specified number of epochs.
- **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus, helping the model converge.

### 4. Hyperparameters

The key hyperparameters for the model are:
- **Epochs**: 40
- **Batch Size**: 16
- **Input Image Size**: 299x299
- **Classes**: 12 (plant species)

### 5. Training the Model

Once the data is preprocessed and augmented, and the model is built, the training process begins. The notebook allows you to train the model by running a cell, and it includes automatic saving of the best model based on validation accuracy.

To start the training, simply run the training cell in the notebook.

### 6. Evaluation and Inference

After training, the model can be evaluated using test data or validation data to assess its accuracy. The notebook provides a section where you can run model evaluation to check its performance.

You can also use the trained model to classify new plant seedling images. The notebook provides a utility to run **inference** on any new image by loading the trained model and predicting the class label.

---

## Model Evaluation

After training the model, the notebook provides a section where you can evaluate its performance on validation or test data. It outputs the classification accuracy and other performance metrics.

To evaluate the model, simply run the corresponding cell in the notebook.

---

## Inference

Once the model is trained, you can use it to predict the species of new plant seedling images. The notebook allows you to upload an image, and it will return the predicted class label along with the confidence score.

To run inference, upload the image directly in the notebook and run the inference cell.

---

## Dependencies

The notebook requires the following Python libraries:
- **TensorFlow** (2.x)
- **Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **OpenCV**

These packages are installed automatically within the Colab environment by running the installation cell.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Plant Seedlings Classification Dataset**: The dataset used in this project is available through Kaggle.
- **ResNet Paper**: He et al., 2015. Deep Residual Learning for Image Recognition.

---

### Notes

- Since the entire project is done in a Google Colab notebook, you can directly interact with the model, modify parameters, and experiment with different configurations within the notebook.
- Make sure to adjust the dataset path if needed when running in your own Colab environment.

---

This README provides a clear guide to using and understanding the Google Colab-based project for plant seedlings classification. It outlines how to set up, train, and evaluate the model, all within the Colab environment.
