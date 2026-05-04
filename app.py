from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
class InputData(BaseModel):
    features:list[float]

app=FastAPI()
model=joblib.load('KNN.pkl')


@app.post('/')
async def cancer(data:InputData)->dict:
    predt=model.predict([data.features])
    return {"result":int(predt[0])}