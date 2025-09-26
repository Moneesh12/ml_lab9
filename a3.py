import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def explain_pipeline_with_lime(dataset_path, sample_size=5000):
    # Load dataset
    data = pd.read_csv(dataset_path)
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[:-1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define base models
    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("gb", GradientBoostingClassifier(random_state=42))
    ]

    # Meta model
    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # Stacking classifier inside a pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("stacking", StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=3,
            n_jobs=-1
        ))
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Build LIME explainer on training data
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=[str(c) for c in np.unique(y)],
        discretize_continuous=True
    )

    
    idx = 0
    instance = X_test[idx]

    
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=pipeline.predict_proba,
        num_features=10
    )

    print("Prediction probabilities:", pipeline.predict_proba([instance])[0])
    exp.show_in_notebook(show_table=True)

# Run function
explain_pipeline_with_lime(r"DCT_mal.csv", sample_size=5000)

