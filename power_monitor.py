"""Power monitoring: Shelly plug + nvidia-smi (via pynvml)."""

import logging
import time
from collections import deque

import httpx
import pynvml

logger = logging.getLogger(__name__)

SHELLY_URL = "http://10.1.1.215/rpc/Shelly.GetStatus"
BASELINE_WINDOW = 300  # seconds of idle samples to average
GPU_IDLE_THRESHOLD = 20  # % utilization — below this counts as idle


class PowerMonitor:
    def __init__(self):
        self.gpu_count = 0
        self.system_base = None
        self._baseline_samples: deque[float] = deque(maxlen=BASELINE_WINDOW // 5)
        self._http = httpx.Client(timeout=2)

    def init_nvml(self):
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        logger.info("NVML initialized: %d GPUs", self.gpu_count)

    def read_shelly(self) -> float:
        try:
            r = self._http.get(SHELLY_URL)
            return r.json()["pm1:0"]["apower"]
        except Exception:
            logger.warning("Shelly read failed")
            return 0.0

    def read_gpu_powers(self) -> dict[int, float]:
        powers = {}
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            powers[i] = mw / 1000.0  # mW → W
        return powers

    def read_gpu_utilizations(self) -> dict[int, int]:
        utils = {}
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils[i] = util.gpu
        return utils

    def update_baseline(self, shelly_w: float, gpu_utils: dict[int, int]):
        all_idle = all(u < GPU_IDLE_THRESHOLD for u in gpu_utils.values())

        if all_idle and shelly_w > 0:
            self._baseline_samples.append(shelly_w)

        if self._baseline_samples:
            avg = sum(self._baseline_samples) / len(self._baseline_samples)
            if self.system_base is None:
                self.system_base = avg
            else:
                # Track minimum observed average, not just current average
                self.system_base = min(self.system_base, avg)

        if self.system_base is None:
            self.system_base = 0.0
