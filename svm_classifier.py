import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Matplotlib backend (fixes display issues on some systems)
matplotlib.use('TkAgg')  # Try 'QtAgg' if 'TkAgg' doesn't work on your system

# Debugging: Check directory and file existence
print("Current Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

# Attempt to read the CSV
try:
    df = pd.read_csv("breast-cancer.csv")  # Make sure file is in the same folder
    print("✅ CSV loaded successfully. Data shape:", df.shape)
    print(df.head())
except Exception as e:
    print("❌ CSV Loading Error:", e)
    exit()

# ⬇️ Data Preprocessing
# Drop non-numeric or irrelevant columns
df = df.drop(['id'], axis=1)

# Convert diagnosis labels from 'M'/'B' to 1/0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select two numerical features for 2D visualization
X = df[['radius_mean', 'texture_mean']]  # You can change these
y = df['diagnosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
clf_linear = svm.SVC(kernel='linear', C=1)
clf_linear.fit(X_train, y_train)
y_pred_linear = clf_linear.predict(X_test)

# RBF SVM
clf_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=1)
clf_rbf.fit(X_train, y_train)
y_pred_rbf = clf_rbf.predict(X_test)

# Accuracy scores
print("\n📊 Accuracy Scores:")
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

# Plotting function
def plot_decision_boundary(clf, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title(title)
    plt.show()

# Plot decision boundaries
print("\n🖼️ Plotting decision boundaries...")
plot_decision_boundary(clf_linear, X, y, "SVM with Linear Kernel")
plot_decision_boundary(clf_rbf, X, y, "SVM with RBF Kernel")
