import sys


class StockPredictionException(Exception):
    def __init__(self, message: str, error: Exception = None):
        self.message = message
        self.error = error
        detail = f"{message}"
        if error:
            _, _, tb = sys.exc_info()
            if tb:
                lineno = tb.tb_lineno
                filename = tb.tb_frame.f_code.co_filename
                detail += f" | File: {filename}, Line: {lineno} | Cause: {str(error)}"
        super().__init__(detail)


class DataIngestionException(StockPredictionException):
    pass


class InsufficientDataException(StockPredictionException):
    pass