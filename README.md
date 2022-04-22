# Automating and Improving Cardiovascular Disease Prediction Using Machine Learning and EMR Data Features from a Regional Healthcare System
This is a hub to host the research of cardiology EHR-based phenotyping disease modeling. 

This work collaborated with Dr. Wael Eid (Email: Wael.eid at usd.edu (WEE))
- Department of Internal Medicine, Division of Endocrinology, St. Elizabeth Physicians Regional Diabetes Center, Covington, Kentucky, United States of America 
- Department of Internal Medicine, University of Kentucky College of Medicine, Lexington, Kentucky, United States of America
- Department of Internal Medicine, Division of Endocrinology, University of South Dakota Sanford School of Medicine, Sioux Falls, South Dakota, United States of America
- Department of Internal Medicine, Division of Endocrinology, Alexandria University, Alexandria, Egypt

## Introduction

Atherosclerotic cardiovascular disease (ASCVD) carries immense health and economic implications, globally.  North America, the Middle East, and Central Asia carry the highest prevalence of CVD, while Eastern Europe and Central Asia have the highest mortality rates attributable to CVD. Risk assessment is a critical step in primary prevention. Current guidelines employ pooled cohort equations (PCE) to calculate the 10-year risk for hard ASCVD. Also various models currently are used in clinical practice to predict ASCVD. However, these models may misclassify risk either by underestimating or overestimating actual observed risk in populations with different comorbidities or with different demographic or socioeconomic determinants. In this study, we developed a clinically-based ASCVD prediction model that can fill the gap in current methodologies.

## Dataset 

We conducted a retrospective, records-based, longitudinal study using datasets from unique EMRs of living patients maintained by a US regional healthcare system. Using a dynamic EMR-based clinical decision-support tool, we used structured query language (SQL) to extract data from records of patients in the St. Elizabeth Health Care System (Kentucky, USA) who had a clinical encounter between January 1, 2009 and April 30, 2020 that involved checking low-density lipoprotein cholesterol (LDL-C). 

Data used in this research were anonymized according to US Health Insurance Portability and Accountability Act (HIPAA) regulations and are available upon reasonable request from the author with support from the St. Elizabeth Healthcare Clinical Research Institute. The study was approved by the St. Elizabeth Health Care Institutional Review Board and a waiver for informed consent was approved, allowing for retrospective data anonymization.

A total of 289,299 inpatient and outpatient records were screened. Records containing pertinent laboratory and vital sign data for the study timeframe were selected to build the ML models (the DataMain group, n = 101,110). Records containing PCE risk scores were assigned to the DataPCE subgroup (n = 54,850). 

![image](https://user-images.githubusercontent.com/98625360/164759975-55aa53c5-09d7-468a-bd49-c915a174c12c.png)
Data is extracted from EMRs and filtered. ML models are built on DataMain, compared with each other for performance, and compared against the PCE risk score for the DataPCE subgroup. Abbreviations: ASCVD, atherosclerotic cardiovascular disease; CS, cross-sectional; EMR, electronic medical record; LT, longitudinal; machine-learning models: RF, random forest; LR, logistic regression; NN, neural networks; NB, naïve Bayes.



## Model

## Citation

## Acknowledgement


