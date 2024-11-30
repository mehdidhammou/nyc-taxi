from pydantic import BaseModel, computed_field
import os
import pandas as pd


class Trip(BaseModel):
    pickup_month: int
    pickup_day: int
    pickup_hour: int

    @computed_field
    @property
    def abnormal_period(self) -> bool:
        path = os.getenv("ABNORMAL_DATES_PATH")
        abnormal_dates = pd.read_csv(path)
        pickup_date = pd.to_datetime(
            f"{self.pickup_month}-{self.pickup_day}", format="%m-%d"
        )
        return pickup_date in abnormal_dates["pickup_date"].values

    def to_pd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pickup_day": [self.pickup_day],
                "pickup_hour": [self.pickup_hour],
                "pickup_month": [self.pickup_month],
                "abnormal_period": [self.abnormal_period],
            }
        )
