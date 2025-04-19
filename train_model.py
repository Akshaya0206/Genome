import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix, classification_report
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


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

y_test_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
labels = le.classes_

print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=labels))
sample_df = df.sample(5, random_state=1).copy()
for idx, row in sample_df.iterrows():
    print(f"\n--- Sample {idx} ---")
    print("P-Values:")
    print(row[pval_cols].to_string())
    print("\nSupporting Pairs:")
    print(row[support_cols].to_string())
    print("\n" + "-" * 30)

true_labels = sample_df['interaction_strength'].values
X_sample = sample_df[support_cols + pval_cols]
X_sample_scaled = scaler.transform(X_sample)
y_sample_pred = model.predict(X_sample_scaled)
y_sample_labels = le.inverse_transform(y_sample_pred)
print("\n--- Predictions vs Actual ---")
for i in range(len(y_sample_labels)):
    print(f"Sample {i+1}: Predicted = {y_sample_labels[i]} | Actual = {true_labels[i]}")
