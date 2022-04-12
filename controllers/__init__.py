REGISTRY = {}

from .basic_controller import BasicMAC
from .traditional_controller import TradMac

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["trad_mac"]=TradMac  # stands for traditional Mac