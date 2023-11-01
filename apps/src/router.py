from fastapi import APIRouter

from apps.src.controller import data_preprocessing_controller, training_controller, prediction_controller, health_controller, config_controller

classifier_router = APIRouter(
    prefix='/classifier',
    tags=['classifier']
)

classifier_router.include_router(data_preprocessing_controller.router)
classifier_router.include_router(training_controller.router)
classifier_router.include_router(prediction_controller.router)
classifier_router.include_router(health_controller.router)
classifier_router.include_router(config_controller.router)
