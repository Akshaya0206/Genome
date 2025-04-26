import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("genome1.xlsx")

columns_to_drop = ['RefSeqName', 'TranscriptName', 'Feature_Chr', 'Annotation', 'Strand',
                   'InteractorName', 'InteractorID', 'Interactor_Chr', 'InteractorAnnotation', 'IntGroup']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


support_cols = ['CG1_SuppPairs', 'CG2_SuppPairs', 'CC1_SuppPairs', 'CC2_SuppPairs', 'CN1_SuppPairs', 'CN2_SuppPairs']
pval_cols = ['CG1_p_value', 'CG2_p_value', 'CC1_p_value', 'CC2_p_value', 'CN1_p_value', 'CN2_p_value']

df['total_supp_pairs'] = df[support_cols].sum(axis=1)
df['avg_p_value'] = df[pval_cols].mean(axis=1)
df['interaction_score'] = df['total_supp_pairs'] / (df['avg_p_value'] + 1e-6)

# Classification function
def classify_strength(score):
    if score >= 1000:
        return 'strong'
    elif score >= 300:
        return 'moderate'
    else:
        return 'weak'

df['interaction_strength'] = df['interaction_score'].apply(classify_strength)
X = df[support_cols + pval_cols]
y = df['interaction_strength']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, 
                                                    random_state=42, stratify=y_encoded)

# Model training
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model and transformers
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Evaluation
y_test_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
labels = le.classes_
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=labels))

# Sample Predictions
sample_df = df.sample(5, random_state=1)
true_labels = sample_df['interaction_strength'].values
X_sample = sample_df[support_cols + pval_cols]
X_sample_scaled = scaler.transform(X_sample)
y_sample_pred = model.predict(X_sample_scaled)
y_sample_labels = le.inverse_transform(y_sample_pred)
print("\n--- Predictions vs Actual ---")
for idx, (i, row) in enumerate(sample_df.iterrows()):
    print(f"\nSample {idx+1}")
    print("P-Values:")
    print(row[pval_cols].to_string(index=True))
    print("Support Pairs:")
    print(row[support_cols].to_string(index=True))
    print(f"Predicted: {y_sample_labels[idx]} | Actual: {true_labels[idx]}")
    print("-" * 30)

#Bar Plot - Actual vs Predicted
labels = le.classes_
actual_counts = np.bincount(y_test, minlength=len(labels))
predicted_counts = np.bincount(y_test_pred, minlength=len(labels))
x = np.arange(len(labels))
width = 0.35  

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
plt.bar(x + width/2, predicted_counts, width, label='Predicted', color='orange')
plt.xlabel("Interaction Strength")
plt.ylabel("Count")
plt.title("Actual vs Predicted Interaction Strength Distribution")
plt.xticks(ticks=x, labels=labels)
plt.legend()
plt.tight_layout()
plt.show()


