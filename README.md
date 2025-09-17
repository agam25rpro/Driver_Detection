# 🚗 Distracted Driver Detection using Deep Learning



An end-to-end computer vision project to classify distracted driving behaviors using transfer learning with EfficientNetB3. This repository contains the complete Jupyter Notebook for data preprocessing, model training, and evaluation, achieving **~89% validation accuracy**.



---



---

## 📝 Project Overview

This project tackles the critical safety issue of distracted driving by leveraging deep learning. Using the State Farm Distracted Driver Dataset, a sophisticated pipeline was developed to accurately identify 10 distinct classes of driver behavior, ranging from texting to safe driving. The core of the project is a convolutional neural network (CNN) built upon the **EfficientNetB3** architecture, fine-tuned to achieve robust and reliable performance. Beyond simple classification, a driver-level risk scoring framework was also integrated to provide practical safety insights.

---

## ✨ Key Features

- **End-to-End Pipeline**: Complete code from data loading and preprocessing to model training and evaluation in a single notebook.
- **High Accuracy**: Achieved **~89% validation accuracy** across 10 behavior classes.
- **Advanced Techniques**: Implemented **transfer learning**, **advanced data augmentation**, and a **staged training strategy** for optimal performance.
- **Robust Validation**: Used **driver-based train-validation splits** to prevent data leakage and ensure the model generalizes well to unseen drivers.
- **Practical Application**: Includes a **driver-level risk scoring framework** to translate model predictions into actionable safety metrics.

---

## 🖼️ Dataset

The project utilizes the public **State Farm Distracted Driver Detection** dataset, which contains thousands of dashboard camera images of drivers. The images are categorized into 10 classes:

- `c0`: **Safe driving**
- `c1`: **Texting - right**
- `c2`: **Talking on the phone - right**
- `c3`: **Texting - left**
- `c4`: **Talking on the phone - left**
- `c5`: **Operating the radio**
- `c6`: **Drinking**
- `c7`: **Reaching behind**
- `c8`: **Hair and makeup**
- `c9`: **Talking to passenger**

You can find the dataset on Kaggle: [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

---

## 🔧 Methodology

The model was developed following these key steps:

1.  **Data Preprocessing**: Images were loaded and associated with their labels. A crucial step was splitting the data based on driver IDs to ensure that images of the same driver did not appear in both the training and validation sets, preventing data leakage.
2.  **Data Augmentation**: Advanced augmentation techniques (e.g., rotation, zoom, brightness adjustments) were applied to the training data to enhance the model's ability to generalize to new, unseen images.
3.  **Transfer Learning**: The pre-trained **EfficientNetB3** model (trained on ImageNet) was used as the backbone. The top classification layers were replaced with new, custom layers suitable for this specific task.
4.  **Staged Training**:
    - **Stage 1**: The base EfficientNetB3 layers were frozen, and only the new custom layers were trained. This allows the model to adapt to the new dataset without destroying the learned features of the base model.
    - **Stage 2**: The entire model (including some of the base layers) was unfrozen and fine-tuned at a very low learning rate, further improving accuracy.

---

## 📊 Results

The model's performance was evaluated on the validation set, achieving:

- **Validation Accuracy**: **~89%**
- **Per-Class Performance**: Strong precision and recall scores across all 10 classes, demonstrating its reliability in distinguishing between subtle actions like texting and operating the radio.

The notebook contains several plots, including the training/validation accuracy curves and a confusion matrix, to visualize the model's performance in detail.



---

## 🛠️ Technologies Used

- **Language**: Python
- **Libraries**:
    - **TensorFlow / Keras**: For building and training the deep learning model.
    - **Pandas & NumPy**: For data manipulation and numerical operations.
    - **Scikit-learn**: For data splitting and performance metrics.
    - **OpenCV**: For image processing.
    - **Matplotlib & Seaborn**: For data visualization and plotting results.
- **Environment**: Jupyter Notebook

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn opencv-python matplotlib seaborn
    ```

3.  **Download the dataset:**
    - Download the data from the [Kaggle competition page](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
    - Unzip the files and place the `imgs` folder in the root directory of this project.

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the `.ipynb` file and run the cells sequentially to execute the entire pipeline.

---

## 📂 Repository Structure

The repository is kept simple, with the entire project contained in a single file:

```
.
└──  distracted_driver_detection.ipynb   # Main notebook with all code, analysis, and plots.
```
