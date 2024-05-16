import os
import logging as lg
from pprint import pformat 
from datetime import datetime


class WriteLogger:
    
    def __init__(
        self, name='my_logger', path=None, 
        filename_preffix='my_process', 
        format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lg.INFO,
        propagate=False
    ):
        self.name = name
        self.filename = None
        if path is not None:
            if filename_preffix is None:
                filename_preffix = 'my_process'
            if not os.path.exists(path):
                raise FileNotFoundError(f'Argument `path` doesn´t exists, add a valid path; path={path}')
            execution_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{filename_preffix}_{execution_timestamp}.log'
            self.filename = os.path.join(path, filename)

        self.logger = lg.getLogger(self.name)
        if propagate is not None:
            if not isinstance(propagate, bool):
                raise TypeError(f'Arg `propagation` must be bool; provided={type(propagate)}')
            # self.logger.propagate = False     # Avoids duplicated messages from child/parent loggers
            self.logger.propagate = propagate

        if self.logger.hasHandlers():
            self.logger.handlers.clear() # Avoids duplicated messages from root logger
            
        self.logger.setLevel(level)
        self.formatter = lg.Formatter(format, datefmt)
        if self.filename is not None:
            handler = lg.FileHandler(self.filename)    
        else:
            handler = lg.StreamHandler()
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def set_level(self, level):
        self.logger.setLevel(level)
        return None

    def exception(self, exception):
        self.logger.exception(exception)

    def __call__(self, *messages, level='info', **pformat_kwargs):
        try:
            func = getattr(self.logger, level)
        except Exception as exception:
            self.logger.exception(exception)
            raise ValueError(f'Cannot write log because `level`:{level} doesnt exists') from exception
        for message in messages:
            if not isinstance(message, str):
                message = pformat(message, **pformat_kwargs)
                message = f'\n{message}'
            func(message)
        return None


class BaseLogger:

    logger = WriteLogger()

    @staticmethod
    def set_base_logger(*args, **kwargs):
        logger = WriteLogger(*args, **kwargs)
        BaseLogger.logger = logger
        return logger

    @staticmethod
    def add_module_logger(name, level=lg.INFO):
        logger = lg.getLogger(name)
        logger.setLevel(level)
        if BaseLogger.logger.filename is not None:
            handler = lg.FileHandler(BaseLogger.logger.filename)
        else:
            handler = lg.StreamHandler()
        handler.setFormatter(BaseLogger.logger.formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def set_logger_level(logger_name, level):
        logger = lg.getLogger(logger_name)
        logger.setLevel(level)
        return None

    def __call__(self, *args, **kwargs):
        return BaseLogger.logger(*args, **kwargs)
