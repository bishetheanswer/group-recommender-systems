import pandas as pd
from surprise import AlgoBase
import aggregation_functions as agg


def recommend_movies_agg_model(model: AlgoBase, group_id: int, n: int) -> list[str]:
    """Recomienda películas a un grupo de usuarios dado un modelo de recomendación
    agregado.

    Args:
        model (AlgoBase): Modelo de recomendación de predicciones agregadas.
        group_id (int): Id del grupo de usuarios.
        n (int): Número de recomendaciones a generar.

    Returns:
        List[str]: Lista de títulos de películas recomendadas.
    """
    movies = get_all_movie_ids()
    predictions = []
    for movie in movies:
        prediction = model.predict(group_id, movie)
        predictions.append(prediction)
    df_predictions = pd.DataFrame(predictions)
    group_recommendations = df_predictions.sort_values(by="est", ascending=False).head(
        n
    )["iid"]
    return movie_ids_to_titles(group_recommendations)


def recommend_movies_agg_predictions(
    model: AlgoBase,
    users_in_group: list[int],
    n: int,
    agg_function: agg.PreferenceAggregationFunction,
) -> list[str]:
    """Recomienda películas a un grupo de usuarios dado un modelo de predicciones
    agregadas.

    Args:
        model (AlgoBase): Modelo de recomendación de predicciones agregadas.
        users_in_group (list[int]): Lista de ids de usuarios en el grupo.
        n (int): Número de recomendaciones a generar.
        agg_function (agg.PreferenceAggregationFunction): Función de agregación a utilizar.

    Returns:
        List[str]: Lista de títulos de películas recomendadas.
    """
    movies = get_all_movie_ids()
    predictions = []
    for user in users_in_group:
        for movie in movies:
            prediction = model.predict(user, movie)
            predictions.append(prediction)

    df_predictions = pd.DataFrame(predictions)
    group_recommendations = agg.get_recommendations_for_group(
        df_predictions, n, agg_function
    )
    return movie_ids_to_titles(group_recommendations)


def get_all_movie_ids() -> list[int]:
    """Devuelve todos los ids de las películas."""
    return pd.read_csv("data/ratings.csv")["movieId"].unique()


def movie_ids_to_titles(movie_ids: list[int]) -> list[str]:
    """Convierte una lista de ids de películas en una lista de títulos."""
    df_movies = pd.read_csv("data/movies.csv")
    return df_movies[df_movies["movieId"].isin(movie_ids)]["title"].tolist()
