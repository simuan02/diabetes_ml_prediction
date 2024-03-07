import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import lime
from lime import lime_tabular
import shap
import matplotlib.pyplot as plt

dataset_path = "dataset.csv"
df = pd.read_csv(dataset_path)

# Dividi il dataset in features (X) e target (y)
X = df[['gender', 'age', 'hypertension', 'BMI', 'glucose', 'smoking_history', 'HbA1c']]
y = df['diabetes']

# Trasforma le feature categoriche in dummy variables
X = pd.get_dummies(X, columns=['gender', 'smoking_history'])

# Divide il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizza le feature per il modello KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addestra il modello KNN
knn_classifier = KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(X_train_scaled, y_train)

# Valuta le performance del modello su test set
y_pred = knn_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = "weighted")

# Calcola e stampa la curva AUC-PRC
precision, recall, _ = precision_recall_curve(y_test, y_pred)
area_under_curve = auc(recall, precision)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-PRC: {area_under_curve:.4f}')

X_small = X_test_scaled[:1000]  # Utilizza solo i primi 1000 record
shap_data = shap.kmeans(X_small, 50)
explainer = shap.KernelExplainer(knn_classifier.predict_proba, shap_data)
shap_values = explainer.shap_values(X_small)

# Genera il summary plot
shap.summary_plot(shap_values, X_small, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], show=False)
plt.title("SHAP Feature Importance")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC-PRC = {area_under_curve:.2f}')
plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

# Utilizza LIME per spiegare le predizioni senza normalizzazione
explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
knn_classifier2 = KNeighborsClassifier(n_neighbors=10)
knn_classifier2.fit(X_train, y_train)
lime_exp = explainer_lime.explain_instance(X_test.iloc[0].values, knn_classifier2.predict_proba, num_features = len(X.columns), num_samples = 10000)

for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight}")
