from fastapi import APIRouter

from apps.src.controller import data_preprocess_controller, train_controller, predict_controller, health_controller

classifier_router = APIRouter(
    prefix='/classifier',
    tags=['classifier']
)

classifier_router.include_router(data_preprocess_controller.router)
classifier_router.include_router(train_controller.router)
classifier_router.include_router(predict_controller.router)
classifier_router.include_router(health_controller.router)
