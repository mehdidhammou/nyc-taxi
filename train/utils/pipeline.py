from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_pipeline(cat_features, num_features) -> Pipeline:
    column_transformer = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("scaling", StandardScaler(), num_features),
        ]
    )

    return Pipeline(
        steps=[
            ("ohe_and_scaling", column_transformer),
            ("regression", Ridge()),
        ]
    )
