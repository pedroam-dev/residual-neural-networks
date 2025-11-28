# FastAPI service to classify an image as a cat or dog
# activate Pyenv: $ source venv/bin/activate
# run locally: $ uvicorn cats_dogs_classification_fastapi:app --reload
# test terminal: $ 
# test browser: http://localhost:8000/docs 
# kill TCP connections on port 8080: sudo lsof -t -i tcp:8000 | xargs kill -9
# Request body example:

from fastapi import FastAPI, File
from predict import load_model, prediction

app = FastAPI(title="Cats and Dogs Classification API", description="API for classify cats and dogs in images")

load_model('../../models/cat-dogs-model.pt')

@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):
    p = prediction(file)
    return p