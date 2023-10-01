from fastapi import APIRouter

router = APIRouter()

@router.post('/train')
def train_controller(train_config):
    pass