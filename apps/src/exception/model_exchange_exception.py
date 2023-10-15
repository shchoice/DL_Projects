class ModelExchangeException(Exception):
    def __init__(self, message="Model exchange error occured", status_code=500):
        super().__init__(message)
        self.status_code=status_code
