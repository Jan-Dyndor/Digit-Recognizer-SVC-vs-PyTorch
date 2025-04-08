
# 🧾 Project Description

This project explores the effectiveness of Support Vector Classifier (with other shallow learning algorythms) and simple Fully Connected Neural Network (FCNN) - Linear nad non Linear - implemented in PyTorch for handwritten digit recognition using the MNIST dataset. The goal is to compare the performance of traditional machine learning algorithms against deep learning approaches in image classification tasks.

## 🎯 Objective

- **Model Comparison**: Evaluate and compare the performance of multiple models on the MNIST dataset, including:  
  1) Support Vector Classifier (SVC),  
  2) a non-linear Fully Connected Neural Network (FCNN),  
  3) a simple linear classifier implemented in PyTorch.  
  Several other models were also tested, but SVC achieved the best overall performance and was selected as the final baseline.
- **Implementation Practice**: Gain hands-on experience in implementing both traditional machine learning models and deep learning architectures using PyTorch.
- **Performance Analysis**: Analyze and compare key metrics such as accuracy, precision, recall, and F1-score to determine the strengths and limitations of each approach.

## 📁 Dataset

- **Name**: MNIST Handwritten Digits
- **Description**: A dataset comprising grayscale images of handwritten digits (0-9), each of size 28x28 pixels.
- **Source**: [MNIST Database](https://www.kaggle.com/competitions/digit-recognizer)
- **Preparation**:
  - Normalization of pixel values to the range [0,1].
  - Flattening of 28x28 images into 784-dimensional vectors in NumPy
  - Reshaping and tensor conversion for PyTorch models.

## 🧠 Models
- **Tested shallow learning models:**:
  - Logistic Regression
  - Random Forest
  - SVC
  - KNN
  - SDG
- **Fully Connected Neural Network (FCNN) in PyTorch**:
  - **Architecture**:
      - Linear Model:
        - Flatten
        - Input Layer: 784 neurons
        - Hidden Layer:  with 128  neurons 
        - Output Layer: 10 neurons 
        - Optimization using Adam optimizer and Cross-Entropy Loss.
        - Trained over 50 epochs with a batch size of 32.
    - Non-Linear Model:
        - Flatten
        - Input Layer: 784 neurons
        - ReLU activation function
        - Hidden Layer:  with 128  neurons 
        - Output Layer: 10 neurons 
        - Optimization using Adam optimizer and Cross-Entropy Loss.
        - Trained over 50 epochs with a batch size of 32.

## 🛠️ Tools and Libraries

- Python
- Scikit-learn
- PyTorch
- NumPy


## 📊 Results


| Metric          | SVC      | Linear Deep Model | Non-Linear Deep Model |
|-----------------|----------|-------------------|------------------------|
| Accuracy        | 0.971399 | 0.877976          | 0.964881               |
| F1 Score        | 0.971175 | 0.876618          | 0.964869               |
| Precision       | 0.971185 | 0.880218          | 0.965182               |
| Recall          | 0.971214 | 0.877368          | 0.964742               |



## 🔍 Observations

- **SVC** achieved the best overall performance across all evaluated metrics, including accuracy (0.9714), F1 score (0.9712), precision (0.9712), and recall (0.9712). This consistency demonstrates that the model is both precise and sensitive in detecting all classes.
- The **non-linear deep model (FCNN)** performed almost on par with SVC, achieving strong results across all metrics (accuracy = 0.9649, F1 = 0.9649). It significantly outperformed the linear model, confirming the importance of non-linearity when modeling complex patterns in image data.
- The **linear deep model**, while still performing decently, lagged behind both SVC and the FCNN. It achieved an accuracy of 0.8780 and F1 score of 0.8766, indicating its limited capacity to capture non-linear structures in the data.

## 📌 Conclusions

- **SVC** remains the top-performing model in this project and serves as a strong baseline for digit classification tasks.
- **Non-linear deep learning models** such as FCNN are highly competitive and offer an attractive alternative to traditional ML approaches, especially when scalability or GPU acceleration is available.
- **Linear models** are simple and fast but lack the expressive power required for image-based tasks like digit recognition.
- Choosing the right model should depend on the task's complexity, available computational resources, and deployment context. In environments where interpretability and deterministic behavior are key, SVC remains a robust choice. For end-to-end deep learning pipelines, FCNN offers great flexibility and near-state-of-the-art performance.


## 🎓 Lessons Learned & Future Directions

- Implementing models provided insights into their respective strengths and limitations in handling image data.
- Future work could explore convolutional neural networks (CNNs) for potentially improved performance on image classification tasks.
- Experimenting with other datasets and more complex architectures could further enhance understanding and model efficacy.

## 🙋‍♂️ Author

**Jan Dyndor**  
ML Engineer & Pharmacist  
📧 dyndorjan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jan-dyndor/)  
📊 [Kaggle](https://www.kaggle.com/jandyndor)  
💻 [GitHub](https://github.com/jandyndor)

## 🧠 Keywords

machine learning, deep learning, SVC, PyTorch, MNIST, digit recognition, image classification, neural networks
