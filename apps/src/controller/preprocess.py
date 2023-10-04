from fastapi import APIRouter, Response, status
from typing import Dict, Any, Type

from apps.src.config import constants
from apps.src.schemas.preprocess_config import PreprocessConfig
from apps.src.utils.yaml.load import load_data_config

router = APIRouter()

@router.post('/preprocess')
def preprocess_controller(preprocess_config: PreprocessConfig, response: Response) -> Dict[str, Any]:
    try:
        data_config = load_data_config(
            yaml_file=constants.CONFIG_DATA_YAML_FILE_NAME,
            schema=preprocess_config
        )

        response.status_code = status.HTTP_200_OK
        return {"Message": "Success"}
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        log_error(e)
        return {"Error": "Request Failed"}