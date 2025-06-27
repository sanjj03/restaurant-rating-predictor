# 🍴 Restaurant Rating Predictor

A simple Streamlit web app that predicts restaurant ratings based on features like:

- Average cost for two
- Table booking (Yes/No)
- Online delivery (Yes/No)
- Price range (1–4)

Built using Python, scikit-learn, and Streamlit.

## 🔧 How to Run

1. Install required libraries  
   `pip install streamlit pandas scikit-learn`

2. Run the app  
   `streamlit run app.py`

## 🧠 Model

- Random Forest Regressor
- Trained on cleaned restaurant data
- Scaled using StandardScaler
