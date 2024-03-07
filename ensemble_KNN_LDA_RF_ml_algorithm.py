import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

# Suddivisione del dataset in training set e test set
X = dataset[['gender', 'smoking_history', 'hypertension', 'BMI', 'HbA1c', 'glucose', 'age']]
y = dataset['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessamento delle feature categoriche
X_train = pd.get_dummies(X_train, columns=['gender', 'smoking_history'])
X_test = pd.get_dummies(X_test, columns=['gender', 'smoking_history'])

# Modelli
knn_model = KNeighborsClassifier(n_neighbors=5)
lda_model = LinearDiscriminantAnalysis()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensemble modello
ensemble_model = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('lda', lda_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' per ottenere le probabilità delle classi

# Addestramento del modello ensemble
ensemble_model.fit(X_train, y_train)

# Valutazione del modello sul test set
predictions = ensemble_model.predict(X_test)
probabilities = ensemble_model.predict_proba(X_test)[:, 1]

# Calcolo e stampa delle metriche di valutazione
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
precision, recall, _ = precision_recall_curve(y_test, probabilities)
auc_prc = auc(recall, precision)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-PRC: {auc_prc:.4f}")

# SHAP Summary Plot con KernelExplainer
"""X_small = X_test.iloc[:300]
shap_data = shap.kmeans(X_small, 30)
explainer = shap.KernelExplainer(ensemble_model.predict_proba, shap_data, nsamples=50)
shap_values = explainer.shap_values(X_small)

shap.summary_plot(shap_values, X_small, feature_names=X_small.columns, class_names=['No Diabetes', 'Diabetes'], show=False)
plt.title("SHAP Feature Importance")
plt.show()"""

# LIME explanations
explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, mode='classification')
lime_explanation = explainer_lime.explain_instance(X_test.values[0], ensemble_model.predict_proba, num_features=X_train.shape[1])
for feature, weight in lime_explanation.as_list():
    print(f"{feature}: {weight}")

# Plot della curva PRC
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()
