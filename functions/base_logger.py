import os
import logging as lg
from pprint import pformat 
from datetime import datetime


class WriteLogger:
    
    def __init__(
        self, 
        name:str='my_logger', 
        path:str=None, 
        filename_preffix:str='my_process', 
        format:str='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
        datefmt:str='%Y-%m-%d %H:%M:%S', 
        level:str=lg.INFO,
        propagate:bool=False
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

    def set_level(self, level:str):
        self.logger.setLevel(level)
        return None

    def exception(self, exception:Exception):
        self.logger.exception(exception)

    def __call__(self, *messages, level:str='info', **pformat_kwargs):
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
