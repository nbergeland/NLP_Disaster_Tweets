# NLP_Disaster_Tweets 
## Overview
This document provides a comprehensive overview of the NLP Disaster Tweets classification system, a machine learning solution designed for a Kaggle competition that classifies tweets as disaster-related or non-disaster-related. The system implements a bag-of-words approach using scikit-learn's CountVectorizer and RidgeClassifier to achieve binary text classification on social media data.
## System Purpose and Scope
The NLP Disaster Tweets system serves as a competition solution for automatically identifying tweets that discuss real disasters versus those that use disaster-related words in non-emergency contexts. The system processes 7,613 labeled training tweets and generates predictions for 3,263 test tweets, achieving F1 scores ranging from 0.576 to 0.645 in cross-validation.
### System Architecture Overview
<img width="966" alt="Screenshot 2025-05-31 at 7 20 30 PM" src="https://github.com/user-attachments/assets/673779de-c6eb-467e-8120-3caa795c0626" />
### Machine Learning Pipeline
<img width="938" alt="Screenshot 2025-05-31 at 7 21 50 PM" src="https://github.com/user-attachments/assets/32ab2d8d-7817-4135-8971-60abbb6c7c0f" />
## Core Components
The system consists of three primary code entities that handle the machine learning workflow:
<img width="710" alt="Screenshot 2025-05-31 at 7 22 52 PM" src="https://github.com/user-attachments/assets/5a8b495a-16f4-4744-99fb-78f3afe61db5" />
### Data Flow and Entity Relationships
<img width="1466" alt="Screenshot 2025-05-31 at 7 23 54 PM" src="https://github.com/user-attachments/assets/c8e3440f-ff71-4fba-835f-e529a2717a99" />
### Performance Characteristics
The system achieves the following performance metrics through 3-fold cross-validation:

F1 Score Range: 0.576 - 0.645
Mean F1 Score: ~0.608
Validation Method: model_selection.cross_val_score() with cv=3 and scoring="f1"
Implementation Variants
