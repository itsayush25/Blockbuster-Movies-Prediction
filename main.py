from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI(title="Blockbuster Movie Prediction API")
try:
    model = joblib.load(
        r"E:\movie blockbuster ml project\movie_blockbuster_model.pkl"
    )
except Exception as error:
    raise RuntimeError(f"Model loading failed: {error}")


class MovieData(BaseModel):
    year: int
    decade: int
    genre: str
    runtime_min: float
    roi: float
    rating: float
    vote_count: int
    popularity: int
    budget_INR: int


@app.get("/")
def home():
    return {
        "message": "Welcome to the Blockbuster Movie Prediction API"
    }


@app.get("/about")
def about():
    return {
        "message": "This API predicts blockbuster movies using Machine Learning"
    }


@app.post("/predict")
def predict_movie(movie: MovieData):
    try:
        input_data = pd.DataFrame([movie.model_dump()])
        prediction = model.predict(input_data)

        return {
            "message": "Prediction successful",
            "prediction": str(prediction[0])
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {error}"
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while predicting the movie."
        )

    #  python -m uvicorn main:app --reload