import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import kaggle
import mlflow
import mlflow.pyfunc


class FbProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        return self.model.predict(future)


def download_kaggle_dataset(
    kaggle_dataset: str = "pratyushakar/rossmann-store-sales",
) -> None:
    api = kaggle.api
    print(api.get_config_value("username"))
    kaggle.api.dataset_download_files(
        kaggle_dataset, path="./", unzip=True, quiet=False
    )


def prep_store_data(
    df: pd.DataFrame, store_id: int = 4, store_open: int = 1
) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    df_store = df[(df["Store"] == store_id) & (df["Open"] == store_open)].reset_index(
        drop=True
    )
    return df_store.sort_values("ds", ascending=True)


def train_predict(
    df: pd.DataFrame, train_fraction: float, seasonality: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.DataFrame, Prophet]:

    # grab split data
    train_index = int(train_fraction * df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]

    # create Prophet model
    model = Prophet(
        yearly_seasonality=seasonality["yearly"],
        weekly_seasonality=seasonality["weekly"],
        daily_seasonality=seasonality["daily"],
        interval_width=0.95,
    )

    # train and predict
    model.fit(df_train)
    predicted = model.predict(df_test)

    # Evaluate metrics
    df_cv = cross_validation(
        model, initial="180 days", period="180 days", horizon="365 days"
    )
    df_p = performance_metrics(df_cv)
    return predicted, df_train, df_test, train_index, df_p, model


if __name__ == "__main__":
    import os

    with mlflow.start_run():

        # If data present, read it in, otherwise, download it
        file_path = "./train.csv"
        if os.path.exists(file_path):
            logging.info("Dataset found, reading into pandas dataframe.")
            df = pd.read_csv(file_path)
        else:
            logging.info("Dataset not found, downloading ...")
            download_kaggle_dataset()
            logging.info("Reading dataset into pandas dataframe.")
            df = pd.read_csv(file_path)

        # Transform dataset in preparation for feeding to Prophet
        df = prep_store_data(df)

        # Define main parameters for modelling
        seasonality = {"yearly": True, "weekly": True, "daily": False}

        # Calculate the relevant dataframes
        predicted, df_train, df_test, train_index, df_p, model = train_predict(
            df=df, train_fraction=0.8, seasonality=seasonality
        )

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        mlflow.pyfunc.log_model("model", python_model=FbProphetWrapper(model))
        print(
            "Logged model with URI: runs:/{run_id}/model".format(
                run_id=mlflow.active_run().info.run_id
            )
        )
