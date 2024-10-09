# Video Quality Prediction Using Ensemble Models and Feature Engineering

This project focuses on predicting video quality metrics (SSIM, PSNR, VMAF) based on various complexity features extracted from video encoding. The models integrate ensemble methods, advanced feature engineering, and regularization techniques to enhance performance and prevent overfitting.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
  - [Regularization](#regularization)
  - [Ensemble Methods](#ensemble-methods)
  - [Feature Engineering](#feature-engineering)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Cross-Validation](#cross-validation)
- [Contributing](#contributing)

## Features
- **Advanced Metrics:** Predicts SSIM, PSNR, and VMAF using a wide range of complexity metrics including Advanced Motion Complexity, DCT Complexity, and Histogram Complexity.
- **Ensemble Methods:** Combines multiple models such as XGBoost, Ridge Regression, and Lasso using stacking for better predictions.
- **Regularization:** Implements L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
- **Feature Engineering:** Includes advanced feature engineering techniques like Polynomial Features to capture interactions between different complexity metrics.
- **Feature Importance:** Uses XGBoost to evaluate which features contribute the most to model predictions.
- **Cross-Validation:** Employs cross-validation for robust evaluation.

## Dataset
The dataset contains the following features:

- **Advanced Motion Complexity**
- **DCT Complexity**
- **Temporal DCT Complexity**
- **Histogram Complexity**
- **Edge Detection Complexity**
- **ORB Feature Complexity**
- **Color Histogram Complexity**
- **Bitrate (kbps)**
- **Resolution (px)**
- **Frame Rate (fps)**
- **CRF**
- **SSIM**
- **PSNR**
- **VMAF**
- **average_framerate**
- **min_framerate**
- **max_framerate**
- **smoothed_frame_rate_variation**

### Target Variables:
- **SSIM**
- **PSNR**
- **VMAF**

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zaki699/video-quality-prediction.git
   cd video-quality-prediction
   ```

	2.	Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

	1.	Prepare your dataset in CSV format. Ensure it includes all required complexity metrics and video quality scores (SSIM, PSNR, VMAF).
	2.	Train the models using the provided script:

  ```bash
  python train_and_predict_quality.py
  ```
	3.	The script will generate predictions for SSIM, PSNR, and VMAF, along with evaluation metrics.

## Model Training

Regularization

We apply both L1 (Lasso) and L2 (Ridge) regularization to avoid overfitting and improve model generalization.

	- Lasso (L1) is used for feature selection by penalizing the absolute values of coefficients.
	- Ridge (L2) penalizes large coefficients and helps when features are highly collinear.

Ensemble Methods

We combine predictions from various models using ensemble techniques:

	- XGBoost and LightGBM are gradient boosting models known for their speed and performance.
	- Stacking Regressor: Combines Ridge, Lasso, and XGBoost models to leverage the strengths of each.

Feature Engineering

	- Polynomial features are used to capture interactions between different metrics like motion and edge detection complexity.
	- Feature scaling ensures that models like Ridge and Lasso are not biased towards features with larger scales.

## Evaluation

The models are evaluated using Mean Squared Error (MSE):

	- SSIM MSE
	- PSNR MSE
	- VMAF MSE

Additionally, the final stacking model’s performance is evaluated and printed.

Feature Importance

XGBoost is used to compute feature importance scores, which help in understanding which features are most significant for predicting video quality.

Example plot for SSIM feature importance:

```bash
plot_feature_importance(xgb_model_ssim.feature_importances_, X.columns, 'XGBoost SSIM')
```

Hyperparameter Tuning

Grid search is performed to optimize hyperparameters such as learning rate, depth of trees, and number of estimators for XGBoost.

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
```

Cross-Validation

Cross-validation is used to evaluate the generalization of the model using different folds of the data.

```python
cv_scores = cross_val_score(ridge_ssim, X_train_scaled, y_ssim_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-cv_scores.mean()}")
```

Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you would like to improve the project.

License

This project is licensed under the MIT License - see the LICENSE file for details.
