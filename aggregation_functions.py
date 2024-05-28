from enum import Enum
import pandas as pd
from collections import Counter


class PreferenceAggregationFunction(Enum):
    BORDA_COUNT = "borda_count"
    AVERAGE = "average"
    ADDITIVE_UTILITARIAN = "additive_utilitarian"
    MULTIPLICATIVE = "multiplicative"
    FAIRNESS = "fairness"
    LEAST_MISERY = "least_misery"
    HIGHEST_FREQUENCY = "highest_frequency"


def get_recommendations_for_group(
    df_predictions_group_members: pd.DataFrame,
    n_items: int,
    aggregation_function: PreferenceAggregationFunction,
) -> list[int]:
    """Devuelve las recomendaciones para un grupo de usuarios.

    Args:
        df_predictions_group_members (pd.DataFrame): Dataframe con las columnas uid,
            iid y est. uid representa el id del usuario, iid el id del item y est la
            valoración estimada.
        n_items (int): Número de items a recomendar.
        aggregation_function (PreferenceAggregationFunction): Función de agregación a
            usar. Esto se usa para agregar las recomendaciones de los usuarios de cada
            grupo.

    Returns:
        list[int]: Lista con los ids de los items recomendados.
    """
    match aggregation_function:
        case PreferenceAggregationFunction.AVERAGE:
            return _average(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.ADDITIVE_UTILITARIAN:
            return _additive_utilitarian(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.MULTIPLICATIVE:
            return _multiplicative(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.BORDA_COUNT:
            return _borda_count(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.FAIRNESS:
            return _fairness(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.LEAST_MISERY:
            return _least_misery(df_predictions_group_members, n_items)
        case PreferenceAggregationFunction.HIGHEST_FREQUENCY:
            return _highest_frequency(df_predictions_group_members, n_items)
        case _:
            raise ValueError("Función de agregación inválida.")


def _highest_frequency(
    df_predictions_group_members: pd.DataFrame, n_items: int
) -> list[int]:
    """Devuelve los items más frecuentes entre las recomendaciones de los usuarios."""
    item_frequency = df_predictions_group_members["iid"].value_counts()
    return (
        item_frequency.sort_values(ascending=False)
        .reset_index()["iid"]
        .tolist()[:n_items]
    )


def _least_misery(
    df_predictions_group_members: pd.DataFrame, n_items: int
) -> list[int]:
    """Devuelve los items con la valoración más baja de cada usuario
    ordenados de mayor a menor."""
    min_ratings = df_predictions_group_members.groupby("iid")["est"].min()
    return (
        min_ratings.sort_values(ascending=False).reset_index()["iid"].tolist()[:n_items]
    )


def _fairness(df_predictions_group_members: pd.DataFrame, n_items: int) -> list[int]:
    """Devuelve el item con la valoración más alta de cada usuario."""
    recommendations = []
    for user in df_predictions_group_members["uid"].unique():
        user_predictions = df_predictions_group_members[
            df_predictions_group_members["uid"] == user
        ]
        top_item = (
            user_predictions.sort_values(by="est", ascending=False)
            .head(1)["iid"]
            .values[0]
        )
        recommendations.append(top_item)
        if len(recommendations) == n_items:
            break
    return recommendations


def _additive_utilitarian(
    df_predictions_group_members: pd.DataFrame, n_items: int
) -> list[int]:
    """Devuelve los items con la suma más alta de valoraciones."""
    return (
        df_predictions_group_members.groupby("iid")["est"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()["iid"]
        .tolist()[:n_items]
    )


def _average(df_predictions_group_members: pd.DataFrame, n_items: int) -> list[int]:
    """Devuelve los items con la media más alta de valoraciones."""
    return (
        df_predictions_group_members.groupby("iid")["est"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()["iid"]
        .tolist()[:n_items]
    )


def _multiplicative(
    df_predictions_group_members: pd.DataFrame, n_items: int
) -> list[int]:
    """Devuelve los items con el producto más alto de valoraciones."""
    return (
        df_predictions_group_members.groupby("iid")["est"]
        .prod()
        .sort_values(ascending=False)
        .reset_index()["iid"]
        .tolist()[:n_items]
    )


def _borda_count(df_predictions_group_members: pd.DataFrame, n_items: int) -> list[int]:
    """Devuelve los items siguiendo la estrategia de Borda Count."""
    borda_count = Counter()

    for user in df_predictions_group_members["uid"].unique():
        user_predictions = df_predictions_group_members[
            df_predictions_group_members["uid"] == user
        ].copy()
        user_predictions["rank"] = user_predictions["est"].rank(ascending=True)
        for _, row in user_predictions.iterrows():
            borda_count[row["iid"]] += row["rank"]

    return [item[0] for item in borda_count.most_common(n_items)]
