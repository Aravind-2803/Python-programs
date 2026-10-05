rows = [
    ">=9,Yes,Very good,Good,Yes",
    ">=8,No,Good,Moderate,Yes",
    ">=9,No,Average,Poor,No",
    "<8,No,Average,Good,No",
    ">=8,Yes,Good,Moderate,Yes",
    ">=9,Yes,Good,Moderate,Yes",
    "<8,Yes,Good,Poor,No",
    ">=9,No,Very good,Good,Yes",
    ">=8,Yes,Good,Good,Yes",
    ">=8,Yes,Average,Good,Yes",
    ">=9,Yes,Very good,Good,Yes",
    ">=8,No,Good,Moderate,Yes",
    ">=9,No,Average,Poor,No",
    "<8,Yes,Average,Good,No",
    ">=8,Yes,Good,Moderate,Yes",
    ">=9,Yes,Average,Moderate,Yes",
    "<8,Yes,Good,Poor,No",
    ">=9,No,Very good,Good,Yes",
    ">=8,Yes,Good,Good,No",
    ">=8,Yes,Average,Good,Yes",
    ">=9,No,Good,Good,Yes",
    ">=8,Yes,Good,Good,Yes",
    ">=8,Yes,Average,Good,Yes",
    ">=9,Yes,Very good,Good,Yes",
    ">=8,No,Good,Moderate,Yes",
    "<8,No,Average,Poor,Yes",
    "<8,Yes,Average,Good,No",
    ">=8,Yes,Good,Moderate,Yes"
]

print(len(rows), "rows typed")

# Create CSV file

import pandas as pd
with open("assess.csv", "w") as f:
    f.write("cgpa,Interactiveness,P_K,Comm_Skills,Job_Offer\n")
    f.write("\n".join(rows) + "\n")
ds = pd.read_csv("assess.csv")
print(ds.head())
print(ds.shape)
print(ds["Job_Offer"].value_counts())
print(pd.crosstab(ds["cgpa"],ds["Job_Offer"]))

# Information Gain

import numpy as np
def entropy(col):
    p = col.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()
total = entropy(ds["Job_Offer"])
print("Entropy of Job_Offer:", round(total, 4))

for a in ["cgpa", "Interactiveness", "P_K", "Comm_Skills"]:
    after = sum(len(g) / len(ds) * entropy(g["Job_Offer"])for _, g in ds.groupby(a))
    gain = total - after
    print(a, "gain =", round(gain, 4))

# Encode categorical values

ds["cgpa"] = ds["cgpa"].map({
    ">=9": 2,
    ">=8": 1,
    "<8": 0
})

ds["Interactiveness"] = ds["Interactiveness"].map({
    "Yes": 1,
    "No": 0
})

ds["P_K"] = ds["P_K"].map({
    "Very good": 2,
    "Good": 1,
    "Average": 0
})

ds["Comm_Skills"] = ds["Comm_Skills"].map({
    "Good": 2,
    "Moderate": 1,
    "Poor": 0
})

ds["Job_Offer"] = ds["Job_Offer"].map({
    "Yes": 1,
    "No": 0
})


print("\nEncoded Dataset:")
print(ds.head())


# --------------------------------------------------
# Check for missing values
# --------------------------------------------------

print("\nMissing Values:")
print(ds.isnull().sum())


# --------------------------------------------------
# Split into training and testing sets
# --------------------------------------------------

from sklearn.model_selection import train_test_split

x = ds.drop("Job_Offer", axis="columns")
y = ds["Job_Offer"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=0
)

print("\nTraining Rows:", len(x_train))
print("Test Rows:", len(x_test))


# Train Decision Tree


from sklearn.tree import DecisionTreeClassifier
dtmodel = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)
dt = dtmodel.fit(x_train, y_train)
print("\nTree depth:", dt.get_depth())
print("Number of Leaves:", dt.get_n_leaves())


# Prediction and Accuracy

from sklearn.metrics import accuracy_score, classification_report
y_pred = dt.predict(x_test)
print("\nPredicted:", y_pred.tolist())
print("Actual:   ", y_test.tolist())
print("Accuracy is:",accuracy_score(y_test, y_pred) * 100)
print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=0))



