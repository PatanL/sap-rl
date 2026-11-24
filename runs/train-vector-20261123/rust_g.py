from pathlib import Path
import sap_engine_rs as native
PIN = Path('/sap/engine-rs-pinned/v31g')
def install():
    native.install(str(PIN/'libsap_engine.so'), PIN/'data')
    return native.RustBattle
class Bridge:
    def __init__(self,audit=None):
        if audit is not None:raise ValueError('Use the pinned g match parity check for battle instrumentation')
        install()
    def battle_class(self):return native.RustBattle
