from pydantic import BaseModel, Field, field_validator


class TaxiInput(BaseModel):

    vendor_id: int = Field(
        ...,
        description="Taxi vendor ID",
        examples=[1]
    )

    pickup_datetime: str = Field(
        ...,
        description="Pickup date and time",
        examples=["2016-06-12T08:30"]
    )

    passenger_count: int = Field(
        ...,
        ge=1,
        le=8,
        description="Number of passengers"
    )

    pickup_longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    pickup_latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    dropoff_longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    dropoff_latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    store_and_fwd_flag: str = Field(
        ...,
        description="Y or N"
    )

    @field_validator("pickup_datetime")
    @classmethod
    def validate_datetime(cls, value):

        from datetime import datetime

        try:
            datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(
                "pickup_datetime must be a valid ISO datetime "
                "such as 2016-06-12T08:30"
            )

        return value

    @field_validator("store_and_fwd_flag")
    @classmethod
    def validate_flag(cls, value):

        value = value.strip().upper()

        if value not in {"Y", "N"}:
            raise ValueError(
                "store_and_fwd_flag must be either 'Y' or 'N'"
            )

        return value


class PredictionResponse(BaseModel):

    success: bool
    duration_seconds: float
    duration_minutes: float
    estimated_fare: float
    distance_km: float
    estimated_speed: float


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    pipeline_loaded: bool