# Phishing URL Detection using Machine Learning

This project demonstrates a machine learning-based approach to detecting phishing websites using only real-time extractable features from URLs. It is built as a lightweight, fast prototype that can serve as a strong foundation for a production-ready anti-phishing tool.

## Overview

* **Goal**: Classify a URL as either "phishing" or "legitimate" based solely on features that can be derived without external APIs or services.
* **Tech Stack**: Python, scikit-learn, pandas, joblib
* **Model**: Random Forest Classifier

## Features Used

The model uses real-time extractable features such as:

* URL length
* Digit-to-length ratio
* Number of 'www' occurrences
* Presence of phishing-related keywords (e.g. 'login', 'secure')
* Longest word in URL path
* Number of "http" references

## Files Included

* `Phishing_URL_Detector_Baseline_Model.ipynb`: Jupyter notebook with full training, evaluation, and prediction pipeline
* `group14.csv`: Sample dataset used for training and evaluation
* `phishing_rf_model.pkl`: Trained Random Forest model saved via joblib
* `scaler.pkl`: StandardScaler used to normalize feature values before prediction
* `requirements.txt`: Python packages needed to run the notebook
* `README.md`: This documentation

## How to Run

### 1. Clone or Download Repository

```bash
git clone https://github.com/muhammadgithubcodetech/phishing-url-detector.git
cd phishing-url-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook Phishing_URL_Detector_Baseline_Model.ipynb
```

### 4. Make Predictions

You can run the final cell in the notebook to enter a URL and receive a prediction:

```python
import joblib
import pandas as pd
from urllib.parse import urlparse

model = joblib.load("phishing_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(url):
    parsed = urlparse(url)
    return pd.DataFrame([{
        'length_url': len(url),
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url),
        'nb_www': url.lower().count("www"),
        'phish_hints': int(any(x in url.lower() for x in ['login', 'secure', 'account', 'update'])),
        'longest_word_path': max((len(p) for p in parsed.path.split("/")), default=0),
        'nb_hyperlinks': url.count("http"),
        'google_index': 0,
        'page_rank': 0,
        'web_traffic': 0,
        'domain_age': 0
    }])

url = input("Enter a URL to check: ")
features = extract_features(url)
scaled = scaler.transform(features)
pred = model.predict(scaled)[0]
prob = model.predict_proba(scaled)[0][1]

print("Prediction:", "Phishing" if pred == 1 else "Legitimate")
print(f"Phishing Probability: {prob:.2f}")
```

---

## Limitations

* Only structural URL features are used (no WHOIS, blacklist, DNS, or website content)
* False positives may occur for long, complex, but legitimate URLs
* Placeholder values are used for features not available in real-time (e.g., page\_rank, domain\_age)

## Next Steps / Future Improvements

To evolve this project for real-world deployment:

1. **Improve Feature Engineering**

   * Integrate WHOIS, domain reputation, blacklist, and traffic APIs
2. **Optimize Model**

   * Try XGBoost, LightGBM, or model ensembling
   * Perform probability calibration and threshold tuning
3. **Deploy as a Web Service**

   * Use Flask or FastAPI to wrap model into an API
   * Optionally build a frontend using Streamlit
4. **Security Enhancements**

   * Sanitize input, log predictions, and validate domains
5. **Monitoring & Retraining**

   * Collect usage data and retrain the model periodically

## Usage for Others

Any user can:

* Download this project
* Load the `.pkl` files (`model`, `scaler`)
* Use the `extract_features()` function on any URL
* Run prediction without retraining the model

This makes it lightweight, fast, and usable for real-time phishing detection.

## License

This project is provided for educational and demonstration purposes.

---

Feel free to reach me out on github, fork the project, raise issues, or contribute suggestions.
