import logging

logging.captureWarnings(True)

LOGGER = logging.getLogger("inmoose")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
LOGGER.handlers.clear()
LOGGER.addHandler(handler)
