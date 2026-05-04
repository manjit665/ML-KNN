from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

from pydantic import BaseModel

class InputData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

app=FastAPI()
model=joblib.load('KNN.pkl')


@app.post('/')
async def cancer(data:InputData)->dict:
    features = list(data.model_dump().values()) 
    predt=model.predict([features])
    return {"result":int(predt[0])}