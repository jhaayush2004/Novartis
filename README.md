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
1. *Data Preprocessing*:
   - Handled missing values and irrelevant columns.
   - Transformed categorical features to numerical representations (e.g., one-hot encoding).
   - Extracted embeddings for textual features using BioBERT.
   - Standardized numerical columns for uniform scaling.

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
-  Ayush Shaurya Jha(Mentor & Lead)
-  Satyam kumar(Lead) 
  
## Acknowledgments
The project utilized insights from academic papers and was powered by an NVIDIA A100 GPU through OLA Krutrim.

## References
- [BioBERT Research Paper](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- [Recruitment Rate Insights](https://trialhub.com/resources/articles/clinical-trial-recruitment-rate-4-things-to-know)
