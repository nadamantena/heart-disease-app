# Heart Disease Prediction Flask App

This is a Flask web application that predicts the risk of heart disease based on user input. The model is trained on a publicly available heart disease dataset and provides an interactive web interface for easy risk assessment.

---

## Project Overview

- **Framework:** Flask (Python)
- **Model:** Machine learning model trained on heart disease data
- **Functionality:**  
  - Collects user health data via a form  
  - Predicts heart disease risk  
  - Displays results with recommendations

---

## Dataset

- **Name:** Heart Disease Dataset (`heart.csv`)
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/arezaei81/heartcsv)  
- **Description:** Contains health metrics and diagnosis information for heart disease prediction.

---

## Adding the Dataset

This repository **does not include the dataset** due to file size.  
To use the app, download `heart.csv` from Kaggle and place it in the project's `data/` folder (create it if needed).

---

## How to Use

1. Clone this repository:
    ```
    git clone https://github.com/yourusername/heart-disease-ml-app.git
    cd heart-disease-ml-app
    ```

2. Download the dataset from Kaggle ([heart.csv](https://www.kaggle.com/datasets/arezaei81/heartcsv)) and place it in a folder named `data` at the root of this project.

3. Install Python dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the Flask app:
    ```
    python app.py
    ```

5. Open your web browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the application.

---

## Project Structure (Simplified)

heart-disease-ml-app/
│
├── app.py
├── requirements.txt
├── Procfile
├── data/
│ └── heart.csv (you place the dataset here)
├── templates/
└── static/

## Citation

If you use this project or the dataset, please cite the original data source:

**Dataset:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/arezaei81/heartcsv)

Dataset author: arezaei81, License: Public Domain (check Kaggle for latest license details).

## Contact

For questions, suggestions, or support, feel free to open an issue on this repository or contact me directly:

- **GitHub:** [https://github.com/nadamantena](https://github.com/nadamantena)
- **Email:** nadamantena@gmail.com

I welcome feedback and contributions!


