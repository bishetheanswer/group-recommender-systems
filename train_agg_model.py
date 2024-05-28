import pandas as pd
import argparse
from surprise import SVD, Dataset, Reader, dump

import numpy as np
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

SEED = 4753
np.random.seed(SEED)


def aggregated_model(test_size: float, top_k: int, relevance_rating_threshold: float):
    """Entrena un modelo de recomendación y evalúa su precisión.

    El modelo se entrena con el dataset de movielens y se usa la estrategia de
    modelo agregado, es decir, se agregan las valoraciones de los usuarios de cada
    grupo y se entrena un modelo usando estas valoraciones.
    """
    print("Leyendo conjunto de datos...")
    df = pd.read_csv("data/ratings.csv")
    df_user_groups = pd.read_csv("data/user_group_mapping.csv")
    df = df.merge(df_user_groups, on="userId")
    # calcula la valoración media de cada grupo para cada película
    df_groups = df.groupby(["groupId", "movieId"])["rating"].mean().reset_index()

    print("Dividiendo conjunto de datos en train y test...")
    df_train, df_test = train_test_split(
        df_groups[["groupId", "movieId", "rating"]],
        test_size=test_size,
        random_state=SEED,
    )

    reader = Reader(rating_scale=(0.5, 5.0))
    trainset = Dataset.load_from_df(df_train, reader)
    testset = Dataset.load_from_df(df_test, reader)

    print("Usando GridSearchCV para encontrar los mejores hiperparámetros...")
    param_grid = {
        "n_epochs": [20, 40, 60],
        "lr_all": [0.001, 0.005, 0.01],
        "reg_all": [0.02, 0.05, 0.1],
        "n_factors": [100, 200, 300],
    }

    gs = GridSearchCV(SVD, param_grid, measures=["mse"], cv=3)
    gs.fit(trainset)
    print(f"Mejor MSE {gs.best_score['mse']} con parámetros: {gs.best_params['mse']}")
    best_model = gs.best_estimator["mse"]
    print("Entrenando el mejor modelo...")
    best_model.fit(trainset.build_full_trainset())

    print("Realizando predicciones sobre el conjunto de test...")
    predictions = best_model.test(testset.build_full_trainset().build_testset())
    df_predictions = pd.DataFrame(predictions)
    print("Evaluando el modelo...")
    precision = get_precision_at_k(df_predictions, top_k, relevance_rating_threshold)
    print(f"Precision@{top_k}: {precision}")
    file_path = "models/svd_agg_model.dump"
    print(f"Guardando el modelo en {file_path} ...")
    dump.dump(file_path, algo=best_model)


def get_precision_at_k(df_predictions: pd.DataFrame, k: int, threshold: float) -> float:
    """Calcula la precisión del modelo para un valor de k.

    Args:
        df_predictions: DataFrame con las predicciones del modelo. Debe contener las
            columnas "uid", "iid", "r_ui" y "est". uid es el id del grupo, iid es el
            id de la película, r_ui es la valoración real y est es la valoración
            estimada por el modelo.
        k: Número de recomendaciones a tener en cuenta.
        threshold: Valor umbral para considerar una película relevante.

    Returns:
        float: Precisión del modelo.
    """
    precision_values = []
    for group_id in df_predictions["uid"].unique():
        group_predictions = df_predictions[df_predictions["uid"] == group_id]
        relevant_items = group_predictions[group_predictions["r_ui"] >= threshold][
            "iid"
        ].tolist()
        group_recommendations = (
            group_predictions.sort_values(by="est", ascending=False)
            .head(k)["iid"]
            .tolist()
        )
        relevant_items_in_recommendation = set(relevant_items).intersection(
            set(group_recommendations)
        )
        precision = len(relevant_items_in_recommendation) / k
        precision_values.append(precision)
    return np.mean(precision_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--relevance_rating_threshold", type=float, default=3.5)
    args = parser.parse_args()

    aggregated_model(
        test_size=args.test_size,
        top_k=args.top_k,
        relevance_rating_threshold=args.relevance_rating_threshold,
    )
