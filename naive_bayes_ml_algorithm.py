import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, auc, precision_recall_curve
import shap
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')

# Definisci le features e la variabile target
X = dataset[['gender', 'age', 'smoking_history', 'hypertension', 'glucose', 'HbA1c', 'BMI']]
y = dataset['diabetes']

X = pd.get_dummies(X, columns=['gender', 'smoking_history'])

# Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Valutazione delle performance
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Calcolo della curva Precision-Recall e AUC-PRC
precision, recall, _ = precision_recall_curve(y_test, y_pred)
area_under_curve_prc = auc(recall, precision)

print(f"Accuracy: {accuracy}")
print(f"F1-score Weighted: {f1_weighted}")
print(f"AUC-PRC: {area_under_curve_prc}")

shap_data = shap.kmeans(X_train, 50)
explainer = shap.KernelExplainer(nb_classifier.predict_proba, shap_data)
shap_values = explainer.shap_values(X_test)

# Stampa del Summary Plot di SHAP
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names = ["No Diabetes", "Diabetes"], show=False)
plt.show()

# LIME
explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['Not Diabetic', 'Diabetic'])
lime_exp = explainer_lime.explain_instance(X_test.iloc[0].values, nb_classifier.predict_proba)
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC-PRC = {area_under_curve_prc:.2f}')
plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()
