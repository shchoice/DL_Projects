from fastapi import APIRouter

deep_learning_router = APIRouter(
    prefix='/dl',
    tags=['dl']
)