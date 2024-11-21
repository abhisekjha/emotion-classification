# emotion-classification

# Arousal and Valence Prediction Model

This project aims to predict the **arousal** and **valence** levels of songs based on extracted features from audio data. A **Random Forest Regressor** model is trained using statistical features of the song's arousal and valence data to predict the emotional tone of the song.

## Steps in the Code

### 1. **Data Loading**
   - The code loads **CSV files** containing the **arousal** and **valence** features for each song. These files are assumed to contain data about songs over different time points.

### 2. **Feature Extraction**
   - A function `extract_summary_statistics` calculates **summary statistics** (mean, median, standard deviation, max, and min) for each song's arousal and valence data. These statistics summarize the emotional tone of the song.

### 3. **Data Preparation**
   - The summary statistics (features) for both **arousal** and **valence** are combined into a single matrix `X`, which will be used as the input for training the models.
   - The **labels** (`y_arousal` and `y_valence`) are created by calculating the **mean** of the arousal and valence values across all time points.

### 4. **Data Splitting**
   - The dataset is split into **training** and **testing** sets, with 80% used for training and 20% for testing.

### 5. **Model Training**
   - Two **Random Forest Regressor** models are trained:
     - One model for **arousal** prediction.
     - One model for **valence** prediction.
   - These models learn to predict the emotional tone of the song based on the extracted features.

### 6. **Model Evaluation**
   - The models are evaluated using **Mean Squared Error (MSE)** to measure the accuracy of the predictions. A lower MSE indicates better performance.

### 7. **Saving the Models**
   - The trained models are saved using **joblib**, allowing them to be reused later without needing to retrain them.

### 8. **Output**
   - The **MSE values** for both models are printed to show how well the models are performing on the test data.

## Requirements
- Python 3.x
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn joblib
```

2. Make sure the arousal.csv and valence.csv files are in the correct directory `(./data/annotations/)`.
3. Run the script:
```python model_training.py
```