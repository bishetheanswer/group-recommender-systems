from collections import defaultdict
import aggregation_functions as agg

from typing import Tuple
import argparse

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, dump
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

SEED = 4753
np.random.seed(SEED)


def aggregated_predictions(
    test_size: float,
    top_k: int,
    relevance_rating_threshold: float,
    relevance_min_ratings: int,
):
    """Entrena un modelo de recomendación y evalúa su precisión.

    El modelo se entrena con el dataset de movielens y se usa la estrategia de
    predicción agregada, es decir, se realizan recomendaciones a cada usuario de cada
    grupo independientemente y se agregan las recomendaciones para generar la lista de
    recomendaciones para el grupo.
    """
    print("Leyendo conjunto de datos...")
    df = pd.read_csv("data/ratings.csv")
    df_user_groups_mapping = pd.read_csv("data/user_group_mapping.csv")
    df = df.merge(df_user_groups_mapping, on="userId")

    print("Dividiendo conjunto de datos en train y test...")
    trainset, testset = train_test_split_for_groups(df, test_size)

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
    predictions = best_model.test(testset)

    df_predictions = pd.DataFrame(predictions)
    print("Evaluando el modelo...")
    precision = evaluate_model(
        df_predictions,
        df_user_groups_mapping,
        n_items=top_k,
        relevance_rating_threshold=relevance_rating_threshold,
        relevance_min_ratings=relevance_min_ratings,
    )
    for agg_function, precision in precision.items():
        print(f"Precisión@{top_k} ({agg_function}): {precision}")

    file_path = "models/svd_agg_pred_model.dump"
    print(f"Guardando el modelo en {file_path} ...")
    dump.dump(file_path, algo=best_model)


def train_test_split_for_groups(
    df: pd.DataFrame, test_size: float
) -> Tuple[Dataset, list[Tuple[int, int, float]]]:
    """Divide el dataframe en train y test sets, asegurándose de que los grupos
    de usuarios estén presentes en ambos sets.

    Args:
        df (pd.DataFrame): Dataframe con las columnas userId, movieId y rating.
        test_size (float): Proporción de datos que se usarán para test.

    Returns:
        Tuple[Dataset, list[Tuple[int, int, float]]]: Trainset y testset.
    """
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for group_id in df["groupId"].unique():
        df_group = df[df["groupId"] == group_id]
        train, test = train_test_split(df_group, test_size=test_size, random_state=SEED)
        df_train = pd.concat([df_train, train])
        df_test = pd.concat([df_test, test])

    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(df_train[["userId", "movieId", "rating"]], reader)
    testset = [tuple(x) for x in df_test[["userId", "movieId", "rating"]].values]

    return train_data, testset


def evaluate_model(
    df_predictions: pd.DataFrame,
    df_user_groups_mapping: pd.DataFrame,
    n_items: int,
    relevance_rating_threshold: float,
    relevance_min_ratings: int,
) -> dict[str, float]:
    """Evalúa el modelo de recomendación. Calcula la precisión para cada grupo
    de usuarios y devuelve la media de todas las precisiones por cada función
    de agregación.

    Args:
        df_predictions (pd.DataFrame): Dataframe con las columnas uid, iid, r_ui y est.
            uid representa el id del usuario, iid el id del item, r_ui la valoración
            real y est la valoración estimada.
        df_user_groups_mapping (pd.DataFrame): Dataframe con las columnas userId y
            groupId. Indica a qué grupo pertenece cada usuario.
        n_items (int): Número de items a recomendar por grupo. También se usa para
            calcular la precisión.
        relevance_rating_threshold (float): Umbral de valoración. Junto con
            relevance_min_ratings se usa para determinar los items relevantes para cada
            grupo.
        relevance_min_ratings (int): Número mínimo de valoraciones por item.
            Junto con relevance_rating_threshold se usa para determinar los items
            relevantes para cada grupo.

    Returns:
        dict[str, float]: Diccionario con las precisiones medias para cada función de
            agregación.
    """
    precision_values = defaultdict(list)
    for group_id in df_user_groups_mapping["groupId"].unique():
        user_ids_in_group = df_user_groups_mapping[
            df_user_groups_mapping["groupId"] == group_id
        ]["userId"].tolist()
        df_predictions_group_members = df_predictions[
            df_predictions["uid"].isin(user_ids_in_group)
        ]
        relevant_items = get_relevant_items_for_group(
            df_predictions_group_members,
            threshold=relevance_rating_threshold,
            min_ratings=relevance_min_ratings,
        )
        for agg_function in agg.PreferenceAggregationFunction:
            group_recommendations = agg.get_recommendations_for_group(
                df_predictions_group_members, n_items, agg_function
            )
            relevant_items_in_recommendations = set(relevant_items).intersection(
                set(group_recommendations)
            )
            precision = len(relevant_items_in_recommendations) / n_items
            precision_values[agg_function.value].append(precision)

    return {
        agg_function: np.mean(values)
        for agg_function, values in precision_values.items()
    }


def get_relevant_items_for_group(
    df: pd.DataFrame, threshold: float, min_ratings: int
) -> list[int]:
    """Devuelve los items que han sido valorados por encima de un umbral
    y que han sido valorados por un número mínimo de usuarios.

    Args:
        df (pd.DataFrame): Dataframe con las columnas uid, iid y r_ui.
            uid representa el id del usuario, iid el id del item y r_ui la valoración
            real.
        threshold (float): Umbral de valoración.
        min_ratings (int): Número mínimo de valoraciones por item.

    Returns:
        list[int]: Lista con los ids de los items relevantes.
    """
    high_rated_items = df[df["r_ui"] >= threshold]
    item_counts = high_rated_items["iid"].value_counts()
    relevant_items = item_counts[item_counts >= min_ratings].index.tolist()
    return relevant_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--relevance_rating_threshold", type=float, default=3.0)
    parser.add_argument("--relevance_min_ratings", type=int, default=2)
    args = parser.parse_args()

    aggregated_predictions(
        test_size=args.test_size,
        top_k=args.top_k,
        relevance_rating_threshold=args.relevance_rating_threshold,
        relevance_min_ratings=args.relevance_min_ratings,
    )
