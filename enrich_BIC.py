import pandas as pd
import scipy.stats as stats

# read clinical data and survival labels
df = pd.read_csv(r"D:\Code\Five_Data\BIC\BREAST_Clinical - 副本2.csv", header=0, index_col=0)
df1 = pd.read_csv(r"D:\Paper  Code\Survival_Label\BIC\DSFS_Survival_Label.csv", header=0, index_col=0)

# merge labels into clinical dataframe
df["Label"] = df1["Labels"]
labels_num = list(set(df['Label']))
print("Cluster labels:", labels_num)

# split dataframe by cluster label
Labels = []
for label in labels_num:
    labels = df[df['Label'] == label]
    Labels.append(labels)

# --------------------------- Chi-square test for pathologic_M -----------------------------
unique_pathologic_M = list(set(Labels[0]['pathologic_M']))
print("Unique pathologic_M values:", unique_pathologic_M)

matrix_pathologic_M = []
for i in range(len(labels_num)):
    row = []
    for j in range(len(unique_pathologic_M)):
        count = Labels[i][Labels[i]['pathologic_M'] == unique_pathologic_M[j]].shape[0]
        row.append(count)
    matrix_pathologic_M.append(row)

chi2, p_value, _, _ = stats.chi2_contingency(matrix_pathologic_M)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# --------------------------- Chi-square test for pathologic_N -----------------------------
unique_pathologic_N = list(set(Labels[0]['pathologic_N']))
print("Unique pathologic_N values:", unique_pathologic_N)

matrix_pathologic_N = []
for i in range(len(labels_num)):
    row = []
    for j in range(len(unique_pathologic_N)):
        count = Labels[i][Labels[i]['pathologic_N'] == unique_pathologic_N[j]].shape[0]
        row.append(count)
    matrix_pathologic_N.append(row)

chi2, p_value, _, _ = stats.chi2_contingency(matrix_pathologic_N)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# --------------------------- Chi-square test for pathologic_T -----------------------------
unique_pathologic_T = list(set(Labels[0]['pathologic_T']))
print("Unique pathologic_T values:", unique_pathologic_T)

matrix_pathologic_T = []
for i in range(len(labels_num)):
    row = []
    for j in range(len(unique_pathologic_T)):
        count = Labels[i][Labels[i]['pathologic_T'] == unique_pathologic_T[j]].shape[0]
        row.append(count)
    matrix_pathologic_T.append(row)

chi2, p_value, _, _ = stats.chi2_contingency(matrix_pathologic_T)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# --------------------------- Chi-square test for pathologic_stage -----------------------------
unique_pathologic_stage = list(set(Labels[0]['pathologic_stage']))
print("Unique pathologic_stage values:", unique_pathologic_stage)

matrix_pathologic_stage = []
for i in range(len(labels_num)):
    row = []
    for j in range(len(unique_pathologic_stage)):
        count = Labels[i][Labels[i]['pathologic_stage'] == unique_pathologic_stage[j]].shape[0]
        row.append(count)
    matrix_pathologic_stage.append(row)

chi2, p_value, _, _ = stats.chi2_contingency(matrix_pathologic_stage)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

# --------------------------- Chi-square test for gender -----------------------------
unique_gender = list(set(Labels[0]['gender']))
print("Unique gender values:", unique_gender)

matrix_gender = []
for i in range(len(labels_num)):
    row = []
    for j in range(len(unique_gender)):
        count = Labels[i][Labels[i]['gender'] == unique_gender[j]].shape[0]
        row.append(count)
    matrix_gender.append(row)

chi2, p_value, _, _ = stats.chi2_contingency(matrix_gender)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)