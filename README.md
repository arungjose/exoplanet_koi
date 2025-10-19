# CYGNUS
### An Exoplanet Detector  

A machine learning project built for the **NASA Space Apps Challenge**, designed to detect potential **exoplanets** using a **Random Forest classifier** trained on NASA’s **Kepler Object of Interest (KOI)** dataset.  

The dataset contains nearly **10,000 celestial objects**, each described by **17 physical and orbital parameters**. The model uses these features to predict whether a given object is likely an **exoplanet** or a **false positive**, inspired by the **classic transit detection method** used in astronomy.

---

## Live Demo  
Try it out here 👉 **[Exoplanet Detector Website](https://exoimg-229165992221.asia-south1.run.app/)**  

You can input the 17 parameters directly on the website and instantly get a prediction on whether the object is an exoplanet or not.  
> ⚠️ *Note: The live demo may be taken down in the future as server availability may change.*

---

## Model Overview  
- **Algorithm:** Random Forest Classifier  
- **Dataset:** NASA Kepler Object of Interest (KOI) dataset  
- **Parameters:** 17 stellar/orbital parameters per object  
- **Goal:** Classify whether a celestial body is a confirmed exoplanet  
- **Model Accuracy:** **93%**

The project includes complete preprocessing, training, and evaluation code, along with a ready-to-use model.

---

## ⚙️ How to Run Locally  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/arungjose/exoplanet_koi.git
cd exoplanet_koi
```

### 2️⃣ Create and activate a virtual environment  
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the demo script  
```bash
python test_model.py
```
This script runs the pre-trained model with a predefined set of parameters for an exoplanet (you can modify them in the script to test your own data).

---

## 🧪 Method Used – The Transit Method  
The project is inspired by the **transit photometry technique**, where periodic dips in a star’s brightness indicate a planet passing in front of it. The AI model learns from these patterns in the Kepler dataset to automate and refine detection accuracy.

---

## 💡 Contribution  
Feel free to **clone**, **experiment**, and **develop** this project further.  
If you improve the model, visualization, or UI — contributions are always welcome.  

**Happy Coding ❤️**