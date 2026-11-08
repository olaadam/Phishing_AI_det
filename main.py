import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.dropna(subset=['URL', 'label'])

X_text = df['URL'].astype(str)
y = df['label'].astype(int)

df['url_length'] = X_text.apply(len)
df['num_dots'] = X_text.apply(lambda x: x.count('.'))
df['num_hyphens'] = X_text.apply(lambda x: x.count('-'))

X_numeric = df[['url_length', 'num_dots', 'num_hyphens']].values

scaler = MinMaxScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
X_tfidf = vectorizer.fit_transform(X_text)

X_combined = hstack([X_tfidf, X_numeric_scaled])

X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb_test = nb_model.predict(X_test)
y_pred_nb_val = nb_model.predict(X_val)

svm_model = LinearSVC(random_state=42, max_iter=10000, dual=False)
svm_model.fit(X_train, y_train)
y_pred_svm_test = svm_model.predict(X_test)
y_pred_svm_val = svm_model.predict(X_val)


def print_metrics(y_true, y_pred, model_name, dataset_name):
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Model: {model_name} | Zbiór: {dataset_name}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred))
    print("\n")


print_metrics(y_test, y_pred_nb_test, "Naive Bayes", "Testowy (30%)")
print_metrics(y_val, y_pred_nb_val, "Naive Bayes", "Walidacyjny (10%)")
print_metrics(y_test, y_pred_svm_test, "SVM", "Testowy (30%)")
print_metrics(y_val, y_pred_svm_val, "SVM", "Walidacyjny (10%)")