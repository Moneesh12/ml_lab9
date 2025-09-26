import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def build_stacking_pipeline():
    # base models
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("gb", GradientBoostingClassifier(random_state=42))
    ]
    # meta model
    meta = LogisticRegression(max_iter=1000, random_state=42)

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=3,
        n_jobs=-1
    )
    # use make_pipeline instead of Pipeline
    return make_pipeline(StandardScaler(), stacking)

def run_pipeline_alt(path, sample_size=5000):
    # read data
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # build pipeline
    model = build_stacking_pipeline()

    # CV scores before final training
    cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)

    # train on training split
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\nAlternative Pipeline with Stacking Classifier")
    print("Accuracy on Test Set:", acc)
    print("CV Accuracy:", cv_scores.mean(), "std-", cv_scores.std())

# run
run_pipeline_alt(r"DCT_mal.csv", sample_size=5000)
