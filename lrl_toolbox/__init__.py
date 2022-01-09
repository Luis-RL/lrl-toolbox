"""Top-level package for LRL Toolbox."""

__author__ = """Luis Rodriguez"""
__email__ = 'luisrodluj@gmail.com'
__version__ = '0.0.1'


import logging
import sys

logger = logging.getLogger("lrl_toolbox")

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
