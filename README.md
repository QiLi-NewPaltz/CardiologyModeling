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

### Cross-sectional Features (CS features)
The 31 CS features included demographics, aggregate risk scores, family history, clinical care group, laboratory values, vital signs, and comorbidities.
The summary of features is as follows:
| Feature |	Description(s) |
|---------| ---------------|
|Demographics|	Age |
|Demographics	| Gender  | 
|Demographics	| Age categories: <30, [30,40), [40,50), [50,55), [55,60), [60,65), [65,70), [70,75), [75,80), >=80 |
|Aggregate risk scores | ASCVD 10-year risk score (PCE) |
|Aggregate risk scores | 	ASCVD 10-year risk score (PCE) categorical, discretized to 3 categories: null value, <5, and ≥5 |
|Aggregate risk scores | 	Numerical score for the family history group of the Dutch Lipid Clinic Network (DLCN)  (0,1) |
|Aggregate risk scores | 	Hierarchical Condition Category Risk Score (Risk Score) |
|Aggregate risk scores | 	Numeric score for the LDL-C group of the DLCN (0,1,3,5,8) |
| Family history (FHX)	|	Family history of any coronary artery disease (FHX-++) |
| Family history (FHX)	|	Family history of premature coronary artery disease (FHX Premature) |
| Family history (FHX)	|	Family history of non-premature coronary artery disease (FHX Non-premature) |
| Clinical care group	|	Current insurance carrier (Carrier) |
| Clinical care group	| Current primary care provider is an employee of the healthcare system where the study is conducted or not (SEP Affiliation) |
| Clinical care group	| 	Have seen endocrinologist in the past or not (Saw Endo) |
| Clinical care group	| 	Patient has account with the MyChart personal health record or not (MyChart) |
| Laboratory values	|	Maximum LDL-C (whether EHR-documented or last estimated pretreatment) ≥190 mg/dL at least twice (LDL-C>190 x2) |
| Laboratory values	|	Maximum LDL-C (whether EHR-documented or last estimated pretreatment) ≥190 mg/dL at least once (LDL-C>190) |
| Laboratory values	|	The last LDL-C reading before a CAD diagnosis, or the last LDL-C reading in absence of CAD LDL (Num Before CAD Avg) | 
| Laboratory values	|	The last Non-HDL-C reading before a CAD diagnosis, or the last Non-HDL-C reading in absence of CAD (Non-HDL-C Num Before CAD Avg) | 
| Laboratory values	|	The last VLDL-C reading before CAD diagnosis, or the last VLDL-C-reading in absence of CAD (VLDL-C-Num-Before-CAD-Avg) |
| Laboratory values	|	Maximum Lp(a) (MAX LPA) |
| Laboratory values	|	Maximum Lp(a) group (whether MAX LPA <29 or >50 or null value) (MAX LPA cat) |
| Vital signs	|	Last mean arterial blood pressure (MAP) reading before a CAD diagnosis, or the last MAP reading in absence of CAD (MAP BEFORE CAD Avg)|
| Vital signs	|	Last systolic arterial blood pressure (SYS) reading before a CAD diagnosis, or the last SYS reading in absence of CAD (SYS BP BEFORE CAD Avg) |
| Vital signs	|	Last diastolic arterial blood pressure (DIA) reading before a CAD diagnosis, or the last DIA reading in absence of CAD (DIA BP BEFORE CAD Avg) |
| Comorbidities	|	Diabetes (T1 or T2) (Yes or No, Number of months of having diabetes before being CAD diagnosis) |
| Comorbidities	|	Hypertension (Yes or No, Number of months of having HTN before CAD diagnosis) |
| Comorbidities	|	Obesity (Yes or No, Number of months of having OB before CAD diagnosis) |


### Longitudinal Features
| Feature |	Description(s) |
|---------| ---------------|
| Minimum (MIN)	| Lowest value of all patient recorded values for a feature |
| Maximum (MAX)	| Highest value of all patient recorded values for a feature |
| Average (MEAN) |	Average value for all recorded values of the feature |
| Readings number (COUNT)	| Number of recorded readings for each measure |
| Reading-time range (TRANGE)  |	Time difference, in days, between the first and last recorded values for the feature |
| Reading-value range (VRANGE) |	Difference between the smallest and largest recorded value for the feature |
| Standard deviation (STDEV) |	Amount of variation between the recorded values for the feature |
| Average reading days (Avg-Test-Day) |	Average time, in days, between consecutive recorded values for the feature   |
| Coefficient of variation (CV) |	Standard measure of dispersion of a probability distribution or frequency distribution |


### Machine Learning Models

## Citation

## Acknowledgement


