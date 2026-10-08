from enum import Enum, auto
from typing import Dict, List, Optional
from .core import HALState, HALError, HALDevice

class DeviceManagerState(Enum):
    UNREGISTERED = auto()
    REGISTERED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    STOPPED = auto()
    FAILED = auto()
    DEINITIALIZED = auto()

class HardwareDeviceManager:
    def __init__(self):
        self.devices: Dict[str, HALDevice] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.registration_order: List[str] = []
        self.state = DeviceManagerState.UNREGISTERED
        
    def register(self, name: str, device: HALDevice, deps: List[str] = None) -> HALError:
        if name in self.devices:
            return HALError.INVALID_CONFIGURATION
        self.devices[name] = device
        self.dependencies[name] = deps if deps else []
        self.registration_order.append(name)
        self.state = DeviceManagerState.REGISTERED
        return HALError.OK
        
    def _get_ordered_devices(self) -> List[str]:
        # Simple deterministic ordering based on dependencies and registration order
        ordered = []
        visited = set()
        
        def visit(n):
            if n in visited: return
            for d in self.dependencies.get(n, []):
                visit(d)
            visited.add(n)
            ordered.append(n)
            
        for name in self.registration_order:
            visit(name)
        return ordered

    def initialize_all(self) -> HALError:
        if self.state not in (DeviceManagerState.REGISTERED, DeviceManagerState.DEINITIALIZED):
            return HALError.INVALID_STATE_TRANSITION
            
        ordered = self._get_ordered_devices()
        for name in ordered:
            if name not in self.devices: return HALError.DEVICE_UNAVAILABLE
            err = self.devices[name].init()
            if err != HALError.OK:
                self.state = DeviceManagerState.FAILED
                return err
                
        self.state = DeviceManagerState.INITIALIZED
        return HALError.OK

    def start_all(self) -> HALError:
        if self.state not in (DeviceManagerState.INITIALIZED, DeviceManagerState.STOPPED):
            return HALError.INVALID_STATE_TRANSITION
            
        ordered = self._get_ordered_devices()
        for name in ordered:
            err = self.devices[name].start()
            if err != HALError.OK:
                self.state = DeviceManagerState.FAILED
                return err
                
        self.state = DeviceManagerState.RUNNING
        return HALError.OK

    def stop_all(self) -> HALError:
        if self.state not in (DeviceManagerState.RUNNING, DeviceManagerState.FAILED, DeviceManagerState.INITIALIZED):
            return HALError.INVALID_STATE_TRANSITION
            
        ordered = reversed(self._get_ordered_devices())
        for name in ordered:
            self.devices[name].stop()
            
        self.state = DeviceManagerState.STOPPED
        return HALError.OK

    def deinitialize_all(self) -> HALError:
        if self.state == DeviceManagerState.RUNNING:
            return HALError.INVALID_STATE_TRANSITION
            
        ordered = reversed(self._get_ordered_devices())
        for name in ordered:
            self.devices[name].deinit()
            
        self.state = DeviceManagerState.DEINITIALIZED
        return HALError.OK
