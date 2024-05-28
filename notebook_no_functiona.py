import pandas as pd
from surprise import dump
import inference
import aggregation_functions as agg

# seleccionamos un grupo de usuarios
df_user_groups = pd.read_csv("data/user_group_mapping.csv")
df_group = df_user_groups[df_user_groups["groupId"] == 9]
users_in_group = df_group["userId"].values
df = pd.read_csv("data/ratings.csv")
group_ratings = df[df["userId"].isin(users_in_group)]
df_movies = pd.read_csv("data/movies.csv")
group_ratings = group_ratings.merge(df_movies, on="movieId")[
    ["userId", "title", "rating"]
]

# obtenemos la matriz de usuario-item
user_item_matrix = group_ratings.pivot(index="userId", columns="title", values="rating")

# mostramos las 5 películas mejor valoradas por cada usuario
for user in group_ratings["userId"].unique():
    user_ratings = group_ratings[group_ratings["userId"] == user]
    top_movies = (
        user_ratings.sort_values(by="rating", ascending=False).head(5)["title"].tolist()
    )
    print(f"Usuario {user} - Top películas:\n{top_movies}\n")

# cargamos los modelos
_, agg_predictions_model = dump.load("models/svd_agg_pred_model.dump")
_, agg_model = dump.load("models/svd_agg_model.dump")

# recomendamos con el modelo de predicciones agregadas con dos functiones
# diferentes de agregación
rec = inference.recommend_movies_agg_predictions(
    agg_predictions_model,
    users_in_group,
    n=10,
    agg_function=agg.PreferenceAggregationFunction.LEAST_MISERY,
)
print(f"Modelo de predicciones agregadas con función de agregación Least Misery:\n{rec}\n")
rec = inference.recommend_movies_agg_predictions(
    agg_predictions_model,
    users_in_group,
    n=10,
    agg_function=agg.PreferenceAggregationFunction.BORDA_COUNT,
)
print(f"Modelo de predicciones agregadas con función de agregación Borda Count:\n{rec}\n")

# recomendamos con el modelo agregado
rec = inference.recommend_movies_agg_model(agg_model, group_id=9, n=10)
print(f"Modelo agregado:\n{rec}\n")
