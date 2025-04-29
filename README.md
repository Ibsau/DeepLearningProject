# Deep Learning Job Recommendation System

This project is a deep learning-based job recommendation system that helps early-career professionals discover job opportunities aligned with their skills and experiences. The system leverages **BERT** embeddings and a **CNN model** to evaluate resume-job compatibility and output match scores, providing personalized job recommendations.

---

## Features

- Upload a resume and select a preferred job type
- Get AI-generated job recommendations with:
  - **Match score** (resume-job relevance)
  - **Hiring probability**
- Uses **BERT** for contextual embedding of job descriptions and resumes
- **CNN** architecture to model semantic similarity
- Real-time feedback on job fit

---

## Architecture Overview

1. **Text Preprocessing**  
   - Clean and tokenize resumes and job descriptions  
   - Generate embeddings using pre-trained **BERT**

2. **Model**  
   - Inputs: BERT embeddings of resumes & jobs  
   - Architecture: 1D Convolutional Neural Network (CNN)  
   - Output: Match score (0–1) indicating how well a job fits the candidate

3. **Evaluation**  
   - Metrics: **Accuracy**, **Precision**, **Recall**, **F1 Score**  
   - Plots training/validation performance curves

## **Training and validation accuracy over epochs (sample):**

  - Epoch 2/300  Train Acc: 69.13%  Val Acc: 72.61% 
  - Epoch 8/300  Train Acc: 96.92%  Val Acc: 77.09% 
  - Epoch 30/300 Train Acc: 99.08%  Val Acc: 76.88%

Model converges well, showing good generalization.

## **Dependencies**

Python ≥ 3.8
PyTorch
Transformers (bert-base-uncased)
Matplotlib, NumPy, Scikit-learn


## **How to Run**

1. Train the model
      - python Train.py
   
3. Evaluate
      - python Evaluate.py
   
5. Visualize training
      - python makeTrainGraph.py
