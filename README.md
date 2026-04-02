

# ♻️ AI-Based Plastic Waste Detection System
AI-powered plastic waste detection system using computer vision and machine learning to identify plastic materials from images or live camera input.

An intelligent **computer vision–based system** designed to detect and classify plastic waste using machine learning and image processing techniques. The project integrates a **Flask web application with a trained AI model** to identify plastic objects from images or live camera input, helping demonstrate how artificial intelligence can assist in **smart waste management and environmental sustainability**.

Plastic pollution is a major global challenge, and automated detection systems can significantly improve recycling processes and waste monitoring. Computer vision models can analyze images and classify objects, enabling efficient identification of plastic waste for environmental applications. ([LearnOpenCV][1])

---

# 📌 Project Overview

The **Plastic Waste Detection System** combines machine learning and a web-based interface to identify plastic materials from visual input. The system captures images through a camera or uploaded files and processes them using trained models to predict plastic presence.

The application demonstrates how **AI, computer vision, and web technologies** can be integrated to build intelligent environmental monitoring systems.

This project was developed as part of an **AI hackathon project focusing on sustainable waste management solutions**.

---

# 🚀 Features

* ♻️ Plastic waste detection using machine learning
* 📷 Real-time detection using a camera
* 🧠 Image classification using trained models
* 🌐 Web interface built using Flask
* 📂 Dataset image capture and model retraining
* ⚡ Fast predictions using computer vision
* 🧩 Modular and easy-to-extend architecture

---

# 🛠️ Tech Stack

### Programming

* Python

### Machine Learning / AI

* TensorFlow / Keras
* OpenCV

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Other Tools

* NumPy
* Image processing libraries

---

# 📂 Project Structure

```
plastic_project
│
├── images/                  # Dataset or sample images
├── templates/               # HTML templates for web interface
├── train_model/             # Model training scripts
│
├── app.py                   # Main Flask application
├── app_https.py             # HTTPS-enabled server
├── capture_images.py        # Image dataset collection script
├── live_capture_retrain.py  # Real-time retraining pipeline
├── live_prediction.py       # Live camera prediction script
├── camera_test.html         # Camera testing interface
│
└── .gitignore               # Ignored files
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/abhinavnethi-ui/plastic_project.git
cd plastic_project
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:5000
```

---

# 📷 System Workflow

1️⃣ Capture or upload an image
2️⃣ The system processes the image using OpenCV
3️⃣ The trained model analyzes the image
4️⃣ The application predicts whether plastic is detected
5️⃣ Results are displayed on the web interface

---

# 🎯 Use Cases

* Smart waste management systems
* Recycling automation
* Environmental monitoring
* AI-based sustainability projects
* Educational computer vision demonstrations

---

# 🌍 Environmental Impact

Plastic waste detection systems help improve recycling processes and reduce environmental pollution. Computer vision models can automate waste identification, making monitoring systems faster and more scalable than manual methods. ([LearnOpenCV][1])

Such technologies can contribute to **smart cities, automated recycling plants, and sustainable infrastructure solutions**.

---

# 📸 Future Improvements

* Deploy the system as a cloud web service
* Improve detection accuracy with larger datasets
* Add multiple waste classification categories
* Integrate IoT sensors for smart waste bins
* Deploy mobile or edge-AI versions

---

# 🤝 Contributions

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

# 👨‍💻 Author

**Abhinav Nethi**

🎓 Student | AI & Software Development Enthusiast

GitHub:
[https://github.com/abhinavnethi-ui](https://github.com/abhinavnethi-ui)

---

# ⭐ Support

If you find this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with others
