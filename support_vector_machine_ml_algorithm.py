import pandas as pd
from lime import lime_tabular
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import shap

# Caricamento del dataset (assumendo che sia in formato CSV)
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Definizione delle features e della variabile target
features = ['gender', 'smoking_history', 'hypertension', 'BMI', 'HbA1c', 'glucose', 'age']
target = 'diabetes'

X = df[features]
y = df[target]

# Conversione delle variabili categoriche in dummy variables
X = pd.get_dummies(X, columns=['gender', 'smoking_history'])

# Divisione del dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Valutazione del modello
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision, recall, _ = precision_recall_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
auc_prc = auc(recall, precision)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-PRC: {auc_prc:.4f}')

# SHAP Summary Plot
'''X_small = X_test.iloc[:200]
shap_data = shap.kmeans(X_small, 20)
explainer = shap.KernelExplainer(svm_model.predict_proba, shap_data)
shap_values = explainer.shap_values(X_small)
shap.summary_plot(shap_values, X_small, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], show=False)
plt.title("SHAP Feature Importance")
plt.show()'''

# LIME Explanations
explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], mode='classification')
lime_exp = explainer_lime.explain_instance(X_test.iloc[0], svm_model.predict_proba)
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC-PRC = {auc_prc:.2f}')
plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()