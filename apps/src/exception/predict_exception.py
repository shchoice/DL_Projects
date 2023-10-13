class PredictException(Exception):
    def __init__(self, message="Predict Error occured", status_code=500):
        super().__init__(message)
        self.status_code=status_code
