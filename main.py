from typing import Union, Optional

from fastapi import FastAPI, Path, Query, HTTPException, status
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/about")
def about():
    return{"Data": "About"}
