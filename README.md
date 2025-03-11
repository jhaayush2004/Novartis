# Predicting Recruitment Rate in Clinical Trials
## National Winning Solution at NEST hackathon out of 32000+ registrations across India.
## Overview
This project addresses the problem of predicting the Study Recruitment Rate (RR) for clinical trials, a critical aspect of the drug development process. By implementing a structured approach leveraging advanced machine learning techniques and large language models, this solution provides actionable insights to optimize clinical trial recruitment strategies.

## Key Highlights
- *Objective*: Predict Recruitment Rate (RR) using structured and textual data.
- *Tools and Frameworks*:
  - *Transformers*: Used BioBERT for extracting semantic embeddings from textual data.
  - *PyTorch*: For GPU-accelerated computations.
  - *Scikit-learn*: For Gradient Boosting Model (GBM) training and evaluation.
  - *Bayesian Optimization*: Hyperparameter tuning using bayes_opt.
  - *Google Colab*: For GPU-enabled computation.
  - *Matplotlib & Seaborn*: Data visualization.
  - *Pandas & NumPy*: Data preprocessing and numerical computations.

## Methodology

1.  *Data Preprocessing*:
    *   Handled missing values and irrelevant columns.
        *   Strategically addressed missing values to prevent data loss, employing techniques like imputation with mean/median for numerical columns or mode for categorical ones, depending on the data distribution, thus preserving valuable information.
        *   For columns with excessive missingness, considered dropping them to avoid introducing bias through imputation, acknowledging the trade-off between data quantity and quality.
    *   Transformed categorical features to numerical representations (e.g., one-hot encoding).
        *   Transformed categorical features into numerical representations using one-hot encoding, creating binary columns for each category and enabling the model to process categorical data effectively.
        *   Applied label encoding for ordinal categorical features, mapping categories to numerical values based on their inherent order, capturing the ordinal relationship for the model.
    *   Extracted embeddings for textual features using BioBERT.
        *   Extracted contextual embeddings for textual features using BioBERT, a pre-trained language model, capturing semantic relationships and nuances within the text data.
        *   Fine-tuned BioBERT embeddings to tailor them specifically to the task, enhancing the model's ability to understand and utilize textual information for improved performance.
    *   Standardized numerical columns for uniform scaling.
        *   Standardized numerical columns using StandardScaler, transforming data to have zero mean and unit variance, ensuring uniform scaling and preventing features with larger ranges from dominating the model.
        *   Alternatively, employed MinMaxScaler to scale numerical features to a specific range (e.g., 0 to 1), preserving the original distribution and handling outliers effectively.
    *   **Outlier Handling:**
        *   Identified and addressed outliers using methods like IQR-based filtering or Z-score analysis, mitigating their impact on model training and improving generalization performance.
        *   Applied transformations like log or square root to reduce the skewness of data caused by outliers, making the data distribution more normal and stabilizing variance.
    *   **Class Imbalance Handling:**
        *   Tackled class imbalance issues using techniques like oversampling the minority class (e.g., SMOTE) or undersampling the majority class, creating a more balanced dataset and preventing the model from being biased towards the majority class.
    *   **Feature Engineering:**
        *   Created new features by combining or transforming existing ones, capturing complex relationships within the data and providing the model with additional predictive signals.
        *   Generated interaction features by multiplying or combining different columns, allowing the model to capture non-linear relationships between features and improve accuracy.
    *   **Data Type Conversion:**
        *   Ensured proper data types for each feature, converting them if necessary (e.g., numeric to category) to optimize memory usage and improve computational efficiency.
    *   **Data Transformation**
        *   Applying different mathematical transformation to see the model performace.



2. *Model Training*:
   - Utilized GBM for robust regression, optimized using Bayesian techniques.
   - Compared results with LightGBM as a benchmark.
   - Applied stratified train-test splitting to maintain data consistency.

3. *Evaluation Metrics*:
   - *Root Mean Square Error (RMSE)*: 0.34
   - *Mean Absolute Error (MAE)*: 0.083
   - *R² Score*: 0.45
   - Utilized SHAP for model explainability and feature importance analysis.

## Results
- *Key Features*:
  - Duration of trial, enrollment, and primary completion time were identified as critical predictors.
- *Insights*:
  - Low RMSE and MAE indicate high accuracy and consistency.
  - Explainable AI techniques like SHAP enhance trust in model predictions.

## Challenges
- *Hardware Limitations*: Limited access to high-performance GPUs restricted experimentation with advanced models like GPT-4.
- *Data Imbalance*: Skewed recruitment rates posed challenges in maintaining generalizability.

## Next Steps
- Explore dynamic feature selection using reinforcement learning.
- Fine-tune larger LLMs like LLaMA-3.3 for improved embeddings.
- Implement continuous learning frameworks for model updates with new data.
- Address temporal dynamics using advanced time-series models.

## Lead Members
-  Ayush Shaurya Jha(Mentor & Lead) - IIIT Ranchi
-  Satyam kumar(Lead) - IIT Kanpur
  
## Acknowledgments
The project utilized insights from academic papers and was powered by an NVIDIA A100 GPU through OLA Krutrim.

## References
- [BioBERT Research Paper](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- [Recruitment Rate Insights](https://trialhub.com/resources/articles/clinical-trial-recruitment-rate-4-things-to-know)
