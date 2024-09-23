# Creatus - AI Model Creator

Creatus is an advanced, user-friendly platform designed to help developers create, train, and download AI models for image classification tasks. With a clean and intuitive interface, users can manage their labels, upload images, train their custom model, and test it. The trained models can be downloaded in both TensorFlow Lite (.tflite) and H5 (.h5) formats, making it easy to deploy on various platforms.

The app is accessible at: [creatus.streamlit.app](https://creatus.streamlit.app)

## Features
1. **Manage Labels**: You can add custom labels and upload images corresponding to each label.
   
2. **Train Model**: Once images are uploaded for at least two labels, Creatus allows you to train a custom CNN (Convolutional Neural Network) model.

3. **Test Model**: After training, you can upload a new image and test the model’s prediction accuracy with confidence levels.

4. **Download Model**: Export your trained model in TensorFlow Lite (.tflite) or H5 (.h5) formats, along with usage code, to integrate into other projects.

5. **Advanced Options**: You can configure the number of epochs for training and choose the model export format.

6. **Progress Tracking**: The model training is displayed with a real-time progress bar to track the training process.

7. **Usage Code**: Along with the model download, Creatus generates usage code in Python to load and make predictions using the trained model.

## How to Use
1. **Add Labels**: In the sidebar, enter the name of a label and click "Add Label".
   
2. **Upload Images**: For each label, upload images that correspond to that category. 

3. **Train the Model**: Once you have uploaded images for at least two labels, click the "Train Model" button. The model will start training, and you can see the progress.

4. **Test the Model**: Upload an image to test the trained model, and view the predicted label with a confidence score.

5. **Download the Model**: After training, choose your desired export format (TensorFlow Lite or H5) and click the "Download Model" button to download the model along with the usage code.

## Advanced Options

### Model Architecture

- **Simple CNN**: A basic Convolutional Neural Network with a few layers.
- **VGG-like**: A deeper CNN architecture inspired by the VGG model, which includes more convolutional layers and dense layers.
- **ResNet-like**: A Residual Network architecture that includes residual blocks to mitigate the vanishing gradient problem.
- **Custom**: Allows you to define the number of convolutional layers, dense layers, and the activation function. This gives you full control over the model architecture.

### Data Augmentation

- **Enable Data Augmentation**: When enabled, the training data is augmented to increase the diversity of the training set.
- **Rotation Range**: The range within which images will be randomly rotated.
- **Zoom Range**: The range within which images will be randomly zoomed.
- **Horizontal Flip**: Whether to randomly flip images horizontally.
- **Vertical Flip**: Whether to randomly flip images vertically.

### Training Options

- **Epochs**: The number of times the entire training dataset is passed through the neural network.
- **Learning Rate**: The rate at which the model updates its weights during training.
- **Batch Size**: The number of samples processed before the model's internal parameters are updated.
- **Early Stopping**: Stops training when a monitored metric has stopped improving.
  - **Patience**: The number of epochs with no improvement after which training will be stopped.

### Optimization Options

- **Optimizer**: The optimization algorithm used to update the model's weights.
  - **Adam**: Adaptive Moment Estimation, a popular optimizer that adapts the learning rate.
  - **SGD**: Stochastic Gradient Descent, a simple but effective optimizer.
  - **RMSprop**: Root Mean Square Propagation, an optimizer that adapts the learning rate based on the average of recent gradients.
- **Momentum**: A parameter used with SGD to accelerate gradients vectors in the right directions, thus leading to faster converging.

### Regularization Options

- **L2 Regularization**: Adds a penalty equal to the sum of the squared values of the weights to the loss function.
  - **L2 Lambda**: The strength of the L2 regularization.
- **Dropout**: Randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting.
  - **Dropout Rate**: The fraction of the input units to drop.

### Advanced Visualization Options

- **Show Model Summary**: Displays the summary of the model architecture.
- **Plot Training History**: Plots the training and validation accuracy and loss over epochs.

### Export Options

- **Export TensorBoard Logs**: Exports logs that can be visualized using TensorBoard for detailed analysis of the training process.

### Theme Customization

- **Theme**: Allows you to choose between Light, Dark, and Custom themes.
- **Primary Color, Secondary Color, Background Color, Text Color**: Customize the colors of the app's theme.

### Usage Instructions and Definitions

- **Batch Size**: The number of samples processed before the model's internal parameters are updated.
- **Epochs**: The number of times the entire training dataset is passed through the neural network.
- **Learning Rate**: The rate at which the model updates its weights during training.

### Reset to Normal User

- **Reset to Normal User**: Allows the developer to reset the app to normal user mode, disabling the advanced options.

### Summary

- **Model Summary**: Displays the architecture of the trained model.
- **Training History**: Plots the training and validation accuracy and loss over epochs.


## Note
Creatus offers more advanced customizability and flexibility compared to beginner-friendly platforms like Teachable Machine. It’s perfect for developers who want more control over their model creation process. However, for those who prefer simplicity, platforms like Teachable Machine might be a better starting point.

## Warning
Occasionally, a "ghosting" effect may occur during code execution due to delays. Don't worry, this is a normal occurrence.

## Credits
Creatus was created by **Pranav Lejith** (Amphibiar).

---

Explore Creatus at [creatus.streamlit.app](https://creatus.streamlit.app)



