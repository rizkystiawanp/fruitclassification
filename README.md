
# 🍓 Fruit Classifier App - MobileNetV2 (Streamlit)

This is a fruit image classification project using **MobileNetV2 transfer learning**, wrapped in a simple and interactive **Streamlit web app**.  
It classifies images into five fruit categories and detects irrelevant inputs as **"Unknown / Not a Fruit"**.

---

## 🧠 Description

The model is built using TensorFlow and Keras, leveraging the pretrained MobileNetV2 architecture.  
With proper image augmentation and training callbacks, it achieves up to **95% accuracy** on the test set.

---

## Dataset Source
```
Dataset is taken from :

Fruits Classification 🍇
By DeepNets

```
<a href="https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification">DATASET LINK</a>

---

## 🚀 Features

- Upload image files (JPG, PNG, etc.)
- Enter a direct image URL
- Capture images using your webcam
- View real-time prediction with confidence score
- Flag low-confidence or non-fruit images as “Unknown”
- See prediction history on the sidebar

---

## 💻 Tech Stack

- Python
- TensorFlow (Keras)
- Streamlit
- MobileNetV2 (Transfer Learning)
- PIL / NumPy / Requests

---

## 📁 Project Structure

```
fruit-classifier-app/
├── app.py                     # Main Streamlit app file
├── model_transfer_learning.h5 # Trained MobileNetV2 model
├── requirements.txt           # Required packages for Hugging Face / local run
├── README.md                  # Project documentation
└── assets/                    # (Optional) Folder for demo images or icons
```

---

## 🔧 Run Locally

Make sure you have Python and Streamlit installed.

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Note

This model was trained only to classify **Apple, Banana, Grape, Mango, and Strawberry**.  
Images outside of this scope will be labeled as **Unknown** based on prediction confidence and class margin.

---

## 📝 Author

Created by [Rizky SP](https://github.com/rizkystiawanp) - 2025  
A practical demo of computer vision and transfer learning deployment with Streamlit.<br>
Try app <a href="https://huggingface.co/spaces/rizkystiawanp/FruitClassification">Huggingface</a> <br>
