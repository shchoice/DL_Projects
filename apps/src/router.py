from fastapi import APIRouter

from apps.src.controller import data_preprocess_controller

classifier_router = APIRouter(
    prefix='/classifier',
    tags=['classifier']
)

classifier_router.include_router(preprocess.router)