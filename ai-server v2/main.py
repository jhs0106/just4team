from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Server Running", "model": "KoCLIP/SAM ready"}