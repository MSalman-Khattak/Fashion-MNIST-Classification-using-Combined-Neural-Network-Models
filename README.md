Fashion MNIST Classification using Combined Neural Network Models
This repository contains PyTorch code for classifying the Fashion MNIST dataset using combined neural network models. The models used here are variations of feedforward neural networks, combined to explore different architectures and approaches.

Models Implemented
FirstModel: A basic feedforward neural network model with ReLU activation. This model consists of several fully connected layers with ReLU activation functions.
SecondModel: Another feedforward neural network with softmax activation. It contains fully connected layers with softmax activation, suitable for classification tasks.
ThirdModel: Similar to the FirstModel, this model has ReLU activation. It aims to provide a different architecture for comparison.
Combined_modelInt: Integration of FirstModel and ThirdModel to form a combined model. This model combines the features learned by both models to improve classification accuracy.
Combined_modelBruh: A combination of SecondModel and Combined_modelInt. This model explores a different combination to leverage the strengths of both models.
Usage
Requirements:
PyTorch
Torchvision
Matplotlib (for visualization, if needed)
Training:
Run train.py to train the models.
Hyperparameters such as learning rate, batch size, and the number of epochs can be adjusted in the script.
Models are trained using Stochastic Gradient Descent (SGD) with CrossEntropyLoss as the loss function.
Testing:
After training, run test.py to evaluate the trained models on the test dataset.
Accuracy and average loss are reported.
Hyperparameters:
Number of Epochs: The number of times to iterate over the dataset.
Batch Size: The number of data samples propagated through the network before the parameters are updated.
Learning Rate: Rate at which the models' parameters are updated during training.
Additional Details
Model Combination: The combination of models is performed by integrating the output features of one model with another, followed by a classification layer.
Loss Function and Optimizer: CrossEntropyLoss is used as the loss function, and Stochastic Gradient Descent (SGD) is used as the optimizer.
Training and Testing: Training and testing loops are provided in separate scripts. Feel free to adjust the epochs or other parameters for better performance.
References
FashionMNIST Dataset: PyTorch documentation for FashionMNIST dataset.
PyTorch Tutorials: Official PyTorch tutorials for learning and understanding PyTorch.
Hyperparameter Tuning Tutorial: Tutorial on hyperparameter tuning with PyTorch.
Feel free to modify the code, experiment with different architectures, or adjust hyperparameters for better performance. If you have any questions or suggestions, feel free to reach out!
