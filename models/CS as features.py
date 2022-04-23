#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# In[2]:


import warnings

warnings.filterwarnings("ignore")

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# # Features

labextra = pd.read_csv('data/features.labs.demo.csv')
labextra = labextra.set_index('pid')

featureslabel_bl= pd.read_csv("labs/features.baseline.demo.csv")
featureslabel_bl = featureslabel_bl.set_index('pid')

featureslabel= pd.read_csv("labs/features.all.demo.csv")
featureslabel = featureslabel.set_index('pid')
featureslabel.shape

targets = featureslabel[['CAD', 'Total_CAD_CVD','Adjusted_Angio', 'PCI_or_CABG','CABG_YN','HAD_PCI_YN', 'CTA_DONE_NOT','CAC_DONE_NOT',
              'ABS_CAC_SCORE_L','CAC_Gr_Obst','CTA_Gr_Obst']]

features = featureslabel[featureslabel.columns.difference(targets.columns)]

features_bl = features[features.columns.difference(labextra.columns)]

feature_a = features.loc[features['ASCVD_10_YR_SCORE'].notnull()]
features_a_bl =features_bl.loc[features_bl['ASCVD_10_YR_SCORE'].notnull()]
target_a = targets.loc[features_a_bl.index]


def CI (bootstrapped_scores):
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    #return ("CI: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
    return (str(confidence_lower)+"\t"+str(confidence_upper))
    
def result(ytest,ypred):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootstrapped0p_scores = []
    bootstrapped0r_scores = []
    bootstrapped0f_scores = []
    bootstrapped1f_scores = []
    bootstrapped1p_scores = []
    bootstrapped1r_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(ypred), len(ypred))
        if len(np.unique(ytest[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(ytest[indices], ypred[indices])
        result = classification_report(ytest[indices],ypred[indices],output_dict=True)
    #    print (str(result['0.0']['precision'])+"\t"+str(result['0.0']['recall'])+"\t"+
    #       str(result['0.0']['f1-score'])+"\t"+str(result['1.0']['precision'])+"\t"+
    #       str(result['1.0']['recall'])+"\t"+str(result['1.0']['f1-score']))
        bootstrapped_scores.append(score)
        bootstrapped0p_scores.append(result['0.0']['precision'])
        bootstrapped0r_scores.append(result['0.0']['recall'])
        bootstrapped1p_scores.append(result['1.0']['precision'])
        bootstrapped1r_scores.append(result['1.0']['recall'])
        bootstrapped0f_scores.append(result['0.0']['f1-score'])
        bootstrapped1f_scores.append(result['1.0']['f1-score'])
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    auc = roc_auc_score(ytest, ypred)
    result = classification_report(ytest,ypred,output_dict=True)
    #    print (str(result['0.0']['precision'])+"\t"+str(result['0.0']['recall'])+"\t"+
    #       str(result['0.0']['f1-score'])+"\t"+str(result['1.0']['precision'])+"\t"+
    #       str(result['1.0']['recall'])+"\t"+str(result['1.0']['f1-score']))
    return (str(auc)+"\t"+CI(bootstrapped_scores)+"\t"+
           str(result['0.0']['precision']) +"\t"+ CI(bootstrapped0p_scores)+"\t"+
           str(result['0.0']['recall']) +"\t"+ CI(bootstrapped0r_scores)+"\t"+
           str(result['0.0']['f1-score']) +"\t"+ CI(bootstrapped0f_scores)+"\t"+
           str(result['1.0']['precision']) +"\t"+ CI(bootstrapped1p_scores)+"\t"+
           str(result['1.0']['recall']) +"\t"+ CI(bootstrapped1r_scores)+"\t"+
           str(result['1.0']['f1-score']) +"\t"+ CI(bootstrapped1f_scores))
    #CI(bootstrappedr_scores)
    
def result_prob(ytest,ypred):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(ypred), len(ypred))
        if len(np.unique(ytest[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(ytest[indices], ypred[indices])
     #    print (str(result['0.0']['precision'])+"\t"+str(result['0.0']['recall'])+"\t"+
    #       str(result['0.0']['f1-score'])+"\t"+str(result['1.0']['precision'])+"\t"+
    #       str(result['1.0']['recall'])+"\t"+str(result['1.0']['f1-score']))
        bootstrapped_scores.append(score)
    
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    auc = roc_auc_score(ytest, ypred)
    #    print (str(result['0.0']['precision'])+"\t"+str(result['0.0']['recall'])+"\t"+
    #       str(result['0.0']['f1-score'])+"\t"+str(result['1.0']['precision'])+"\t"+
    #       str(result['1.0']['recall'])+"\t"+str(result['1.0']['f1-score']))
    return (str(auc)+"\t"+CI(bootstrapped_scores))
    #CI(bootstrappedr_scores)


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):

def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


alpha = .95
y_pred = y_pred_lr_cad_cvd#np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
y_true = y_cad_cvd_test#np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

auc, auc_cov = delong_roc_variance(y_true, y_pred)

auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std)

ci[ci > 1] = 1

print('AUC:', auc)
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci)


# # Model Comparision

# ## ROC Curve

y = targets['Total_CAD_CVD']
y = y.replace(np.nan, 0)

X = features
X = X.replace(np.nan, 0)

X_bl = features_bl
X_bl = X_bl.replace(np.nan, 0)

X_a = feature_a
X_a = X_a.replace(np.nan, 0)


X_bl_a = features_a_bl
X_bl_a = X_bl_a.replace(np.nan, 0)



y_a = y[feature_a.index]
y_a = y_a.replace(np.nan, 0)



nb_prediction_proba = nb_clf.predict_proba(X_a)[:, 1]
rf_prediction_proba = rf_clf.predict_proba(X_a)[:, 1]
mlp_prediction_proba = mlp_clf.predict_proba(X_a)[:, 1]
lr_prediction_proba = lr_cad_cvd.predict_proba(X_a)[:, 1]

nb_bl_prediction_proba = nb_bl_clf.predict_proba(X_bl_a)[:, 1]
rf_bl_prediction_proba = rf_bl_clf.predict_proba(X_bl_a)[:, 1]
mlp_bl_prediction_proba = mlp_bl_clf.predict_proba(X_bl_a)[:, 1]
lr_bl_prediction_proba = lr_bl_cad_cvd.predict_proba(X_bl_a)[:, 1]

np.savetxt('data/y_cad_cvd_test.csv', y_a)
np.savetxt('data/ASCVD_10_YR_SCORE.csv', ascvd)

np.savetxt('data/nb_prediction_proba.csv', nb_prediction_proba)
np.savetxt('data/rf_prediction_proba.csv', rf_prediction_proba)
np.savetxt('data/mlp_prediction_proba.csv', mlp_prediction_proba)
np.savetxt('data/lr_prediction_proba.csv', lr_prediction_proba)
np.savetxt('data/nb_bl_prediction_proba.csv', nb_bl_prediction_proba)
np.savetxt('data/rf_bl_prediction_proba.csv',rf_bl_prediction_proba)
np.savetxt('data/mlp_bl_prediction_proba.csv', mlp_bl_prediction_proba)
np.savetxt('data/lr_bl_prediciton_proba.csv', lr_bl_prediction_proba)

nb_noPCE_proba = nb_clf_noPCE.predict_proba(X_a_noPCE)[:, 1]
rf_noPCE_proba = rf_clf_noPCE.predict_proba(X_a_noPCE)[:, 1]
mlp_noPCE_proba = mlp_clf_noPCE.predict_proba(X_a_noPCE)[:, 1]
nb_bl_noPCE_proba = nb_bl_clf_noPCE.predict_proba(X_bl_a_noPCE)[:, 1]
rf_bl_noPCE_proba = rf_bl_clf_noPCE.predict_proba(X_bl_a_noPCE)[:, 1]
mlp_bl_noPCE_proba = mlp_bl_clf_noPCE.predict_proba(X_bl_a_noPCE)[:, 1]

np.savetxt('data/nb_noPCE_proba.csv', nb_noPCE_proba)
np.savetxt('data/rf_noPCE_proba.csv', rf_noPCE_proba)
np.savetxt('data/mlp_noPCE_proba.csv', mlp_noPCE_proba)
np.savetxt('data/nb_bl_noPCE_proba.csv', nb_bl_noPCE_proba)
np.savetxt('data/rf_bl_noPCE_proba.csv',rf_bl_noPCE_proba)
np.savetxt('data/mlp_bl_noPCE_proba.csv', mlp_bl_noPCE_proba)


lr_noPCE_proba = lr_clf_noPCE.predict_proba(X_a_noPCE)[:, 1]
lr_bl_noPCE_proba = lr_bl_noPCE.predict_proba(X_bl_a_noPCE)[:, 1]
np.savetxt('data/lr_bl_noPCE_proba.csv', lr_bl_noPCE_proba)
np.savetxt('data/ lr_noPCE_proba.csv', lr_noPCE_proba)

# ### DataPCE With PCE score as feature

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_cad_cvd_train, y_cad_cvd_test = train_test_split(X_a, y_a, random_state=66,test_size=0.1)
X_bl_train = X_bl_a.loc[X_train.index]
X_bl_test = X_bl_a.loc[X_test.index]
y_bl_cad_cvd_train = y_a.loc[y_cad_cvd_train.index]
y_bl_cad_cvd_test = y_a.loc[y_cad_cvd_test.index]
    
ascvd = feature_a['ASCVD_10_YR_SCORE']/100
ascvd_test = ascvd.loc[y_cad_cvd_test.index]

# Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_cad_cvd_train)
nb_prediction_proba = nb_clf.predict_proba(X_test)[:, 1]


# Ranodm Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=20)
rf_clf.fit(X_train, y_cad_cvd_train)
rf_prediction_proba = rf_clf.predict_proba(X_test)[:, 1]


# Multi-layer Perceptron Classifier
mlp_clf = MLPClassifier(alpha=1, hidden_layer_sizes=150)
mlp_clf.fit(X_train, y_cad_cvd_train)
mlp_prediction_proba = mlp_clf.predict_proba(X_test)[:, 1]


#LR
lr_cad_cvd= LogisticRegression(random_state=0).fit(X_train, y_cad_cvd_train)
lr_prediciton_proba = lr_cad_cvd.predict_proba(X_test)[:,1]
lr_prediciton = lr_cad_cvd.predict(X_test)


# Naive Bayes Classifier
nb_bl_clf = GaussianNB()
nb_bl_clf.fit(X_bl_train, y_bl_cad_cvd_train)
nb_bl_prediction_proba = nb_bl_clf.predict_proba(X_bl_test)[:, 1]


# Ranodm Forest Classifier
rf_bl_clf = RandomForestClassifier(n_estimators=20)
rf_bl_clf.fit(X_bl_train, y_bl_cad_cvd_train)
rf_bl_prediction_proba = rf_bl_clf.predict_proba(X_bl_test)[:, 1]


# Multi-layer Perceptron Classifier
mlp_bl_clf = MLPClassifier(alpha=1, hidden_layer_sizes=150)
mlp_bl_clf.fit(X_bl_train, y_bl_cad_cvd_train)
mlp_bl_prediction_proba = mlp_bl_clf.predict_proba(X_bl_test)[:, 1]




lr_bl_cad_cvd= LogisticRegression(random_state=0).fit(X_bl_train, y_bl_cad_cvd_train)
lr_bl_prediciton_proba = lr_bl_cad_cvd.predict_proba(X_bl_test)[:,1]
lr_bl_prediciton = lr_bl_cad_cvd.predict(X_bl_test)




def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 14})
plt.grid()

fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, lr_prediciton_proba)
plt.plot(fpr, tpr, color='blue', lw=2,  label='LR_LTC_AUC={0:.3f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, rf_prediction_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='RF_LTC_AUC={0:.3f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, mlp_prediction_proba)
plt.plot(fpr, tpr, color='crimson', lw=2,  label='NN_LTC_AUC={0:.3f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, nb_prediction_proba)
plt.plot(fpr, tpr, color='green', lw=2, label='NB_LTC_AUC={0:.3f}'.format(roc_auc))


fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, lr_bl_prediciton_proba )
plt.plot(fpr, tpr, color='royalblue', lw=2,  label='LR_CS_AUC={0:.3f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, rf_bl_prediction_proba)
plt.plot(fpr, tpr, color='orange', lw=2, label='RF_CS_AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, mlp_bl_prediction_proba)
plt.plot(fpr, tpr, color='lightcoral', lw=2,  label='NN_CS_AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_cad_cvd_test, nb_bl_prediction_proba)
plt.plot(fpr, tpr, color='lightgreen', lw=2, label='NB_CS_AUC={0:.3f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(target_a['Total_CAD_CVD'], feature_a['ASCVD_10_YR_SCORE']/100)
plt.plot(fpr, tpr, color='green', lw=2, label='PCE_AUC={0:.3f}'.format(roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.savefig("data/figure6.jpg", dpi=1200)



