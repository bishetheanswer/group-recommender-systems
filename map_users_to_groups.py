import numpy as np
import pandas as pd

SEED = 4753
np.random.seed(SEED)


def create_user_group_mapping(user_ids: list[int], group_size: int) -> pd.DataFrame:
    """Divide a los usuarios en grupos y devuelve un dataframe indicando
    a qué grupo pertenece cada usuario.

    Args:
        user_ids (list[int]): Lista de ids de usuarios.
        group_size (int): Tamaño de los grupos.

    Returns:
        pd.DataFrame: Dataframe con dos columnas: userId y groupId.
    """
    np.random.shuffle(user_ids)
    user_groups = np.array_split(user_ids, len(user_ids) // group_size)
    user_group_mapping = []
    for group_id, group in enumerate(user_groups):
        for user_id in group:
            user_group_mapping.append((user_id, group_id))
    df_user_group = pd.DataFrame(user_group_mapping, columns=["userId", "groupId"])
    return df_user_group


if __name__ == "__main__":
    df = pd.read_csv("data/ratings.csv")
    df_user_group_mapping = create_user_group_mapping(
        df["userId"].unique(), group_size=10
    )
    df_user_group_mapping.to_csv("data/user_group_mapping.csv", index=False)
