from contextlib import contextmanager
import time
import logging


# define a logging wrapper
@contextmanager
def log(message, *args, **kwargs):
    """
    Create a context manger that will log the start and the end of a block of code with the exact time
    it took to finish it
    
    :param process_description: a message in a printf format
    :param args: the arguments for formatting the msg 
    :param kwargs: the only keyword argument allowed is logger to use a specific logger
    :return: 
    """

    if 'logger' in kwargs:
        logger = kwargs['logger']
    else:
        logger = logging.getLogger('context.log')

    start_time = time.time()
    logger.info("Started - " + message, *args)
    yield
    end_time = time.time()
    logger.info("Finished (took: {:.4f} seconds) - ".format(end_time - start_time) + message, *args)

