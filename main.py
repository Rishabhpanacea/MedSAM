import uvicorn
from fastapi import FastAPI

from src.routers import predict_router


from starlette.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)