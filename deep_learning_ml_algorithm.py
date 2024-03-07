import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

X = dataset[['gender', 'smoking_history', 'hypertension', 'BMI', 'HbA1c', 'glucose', 'age']]
y = dataset['diabetes']

# Conversione delle feature categoriche in one-hot encoding
X = pd.get_dummies(X, columns=['gender', 'smoking_history'])

# Suddivisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Definizione del modello in TensorFlow con attivazione softmax nell'output
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(2, activation='softmax')  # Due classi: Diabete e No Diabete
])

# Compilazione del modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Valutazione del modello sul test set
probabilities = model.predict(X_test)
predictions = probabilities.argmax(axis=1)  # Prendi la classe con la probabilità più alta

# Calcolo e stampa delle metriche di valutazione
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
precision, recall, _ = precision_recall_curve(y_test, probabilities[:, 1])  # Probabilità per la classe "Diabete"
auc_prc = auc(recall, precision)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-PRC: {auc_prc:.4f}")

# Plot della curva PRC
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

shap_data = shap.kmeans(X_test, 30)
explainer = shap.KernelExplainer(model.predict, shap_data, nsamples=50)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], show=False)
plt.title("SHAP Feature Importance")
plt.show()

# LIME explanations
explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, mode='classification')
lime_explanation = explainer_lime.explain_instance(X_test.iloc[0], model.predict, num_features=X_train.shape[1])
for feature, weight in lime_explanation.as_list():
    print(f"{feature}: {weight}")
