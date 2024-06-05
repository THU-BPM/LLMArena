from fastapi import FastAPI
import json
import uvicorn

app = FastAPI()

@app.get("/get_json")
async def get_json():
    with open("result.json", "r") as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8245, reload=True)

