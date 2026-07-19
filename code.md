# Rising Waters: A Machine Learning Approach to Flood Prediction

A flood-risk prediction system: Python/scikit-learn models trained on
rainfall and meteorological features, served through a Flask web app.

## Project structure

```
flood_prediction/
├── data/
│   ├── generate_dataset.py   # builds a synthetic flood dataset (swap in a real Kaggle CSV if you have one)
│   └── flood_data.csv        # generated dataset
├── notebooks/eda_plots/      # saved EDA charts (distributions, boxplots, heatmap)
├── models/
│   └── floods.save           # trained model + scaler + label encoder (pickle)
├── templates/                # home, predict, result_flood, result_no_flood
├── static/css/style.css
├── train_model.py            # EDA + preprocessing + model comparison + save best model
├── app.py                    # Flask application
└── requirements.txt
```

## 1. Environment setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `xgboost`. This sandbox had no internet access, so
the model was trained here with `xgboost` unavailable — `train_model.py`
automatically falls back to scikit-learn's `GradientBoostingClassifier` in
that case and prints which one it used. Once you `pip install xgboost`
locally and re-run `train_model.py`, it'll use real XGBoost automatically —
no code changes needed.

## 2. (Optional) Use a real dataset

If you have an actual Kaggle flood dataset, replace `data/flood_data.csv`
with it, keeping these column names (or edit `train_model.py`'s
`feature_cols` to match yours):

`annual_rainfall, seasonal_rainfall, cloud_visibility, humidity, temperature, wind_speed, river_level, drainage_quality, flood`

Otherwise, `python3 data/generate_dataset.py` regenerates the synthetic one.

## 3. Train the models

```bash
python3 train_model.py
```

This runs EDA (saves plots to `notebooks/eda_plots/`), cleans and scales the
data, trains Decision Tree / Random Forest / KNN / XGBoost, prints a
confusion matrix + classification report for each, and saves the best
performer to `models/floods.save`.

## 4. Run the web app

```bash
python3 app.py
```

Visit `http://localhost:5000`. Pages: Home → Run Prediction (form) → Flood
Chance Result or No Flood Chance Result, depending on the model's output.

## 5. Deploying to IBM Cloud

Push this repo to IBM Cloud Foundry or Code Engine as a standard Flask app —
add a `Procfile` with `web: python app.py` and set the port from the
`PORT` env var (`app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))`).

## Notes on this build

- The dataset here is **synthetic** (generated in `data/generate_dataset.py`)
  since no dataset file was uploaded — swap in the real Kaggle data for
  production numbers. The current best model (K-Nearest Neighbors, since
  XGBoost wasn't installable in this sandbox) reaches ~75% accuracy on the
  synthetic data; expect different numbers on your real dataset once you drop
  it in.
