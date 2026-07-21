from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = data.data
y = data.target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='rbf', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
tree_model = DecisionTreeClassifier(random_state=42)
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
nb_model = GaussianNB()
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)

models = {
    "SVM": svm_model,
    "KNN": knn_model,
    "Decision Tree": tree_model,
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "AdaBoost": ada_model,
    "Naive Bayes": nb_model,
    "Neural Network (MLP)": mlp_model
}

print("="*90)
print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("="*90)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"{name:<25} | {acc:.4f}     | {precision:.4f}   | {recall:.4f}   | {f1:.4f}")

print("="*90)
print("\nDetailed Classification Report for Each Model:")
print("="*90)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    print("-"*90)