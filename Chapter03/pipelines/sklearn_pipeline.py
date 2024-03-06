from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

numeric_features = ["age", "balance"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = [
    "job",
    "marital",
    "education",
    "contact",
    "housing",
    "loan",
    "default",
    "day",
]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# Add classifier to the preprocessing pipeline
clf_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

df = pd.read_csv("pipelines/data/bank/bank.csv", delimiter=";", decimal=",")
# Assume there was some EDA and feature analysis to select below
feature_cols = [
    "age",
    "balance",
    "job",
    "marital",
    "education",
    "contact",
    "housing",
    "loan",
    "default",
    "day",
]

# Features and target
X = df[feature_cols].copy()
y = df["y"].apply(lambda x: 1 if x == "yes" else 0).copy()

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf_pipeline.fit(X_train, y_train)
