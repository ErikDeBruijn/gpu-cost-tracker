"""GPU job cost tracking service.

Continuously monitors system power (Shelly) and per-GPU power (nvidia-smi),
fetches EPEX day-ahead prices from ENTSO-E, and exposes a simple API for
workers to track job energy costs.

Attribution model:
- system_base = lowest observed Shelly reading (always-on cost, excluded)
- GPU power from nvidia-smi is directly attributable per GPU
- overhead = shelly_total - system_base - sum(gpu_powers)
- overhead distributed proportionally by GPU power draw
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from epex_prices import EPEXPriceTracker
from power_monitor import PowerMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 5  # seconds


@dataclass
class JobSample:
    timestamp: float
    gpu_power_w: float
    overhead_share_w: float
    total_attributed_w: float
    price_eur_per_kwh: float


@dataclass
class ActiveJob:
    job_id: str
    gpu_index: int
    pid: int | None
    client: str | None
    label: str | None
    started_at: float
    samples: list[JobSample] = field(default_factory=list)


class JobStartRequest(BaseModel):
    gpu: int
    pid: int | None = None
    client: str | None = None  # e.g. "autoresearcher", "ollama"
    label: str | None = None   # e.g. "train_run_42", "llama3-inference"


class JobStopRequest(BaseModel):
    job_id: str


app = FastAPI(
    title="GPU Cost Tracker",
    description="Tracks real energy cost of GPU jobs using Shelly power monitoring, NVML, and EPEX spot prices.",
    version="0.1.0",
)


@app.get("/", response_class=PlainTextResponse)
def root():
    return """GPU Cost Tracker API
====================

Track the real energy cost (kWh + EUR) of GPU jobs.

Endpoints:
  POST /job/start   - Start tracking a job: {"gpu": 0, "client": "myapp", "label": "run_42"}
                      Returns: {"job_id": "abc123"}

  POST /job/stop    - Stop tracking: {"job_id": "abc123"}
                      Returns: {"energy_kwh": 0.12, "cost_eur": 0.03, "duration_s": 300, "avg_power_w": 280}

  GET  /jobs        - Live power + cost for all active jobs (for UI polling)
  GET  /status      - Full system state: Shelly power, GPU power/util, baseline, EPEX price, active jobs
  GET  /openapi.json - OpenAPI schema (machine-readable)
  GET  /docs        - Interactive API docs (Swagger UI)

Attribution model:
  - System baseline (idle power) is excluded — you only pay for what the job causes
  - GPU power is directly attributed per GPU via NVML
  - System overhead (CPU, fans) is split proportionally by GPU power draw
  - Cost = energy * (EPEX spot + 0.125 taxes + 0.02 purchasing) EUR/kWh
"""
monitor = PowerMonitor()
price_tracker = EPEXPriceTracker()
active_jobs: dict[str, ActiveJob] = {}
lock = threading.Lock()


def compute_job_costs(job: ActiveJob) -> dict:
    if not job.samples:
        return {"energy_kwh": 0.0, "cost_eur": 0.0, "duration_s": 0.0, "avg_power_w": 0.0}

    total_energy_wh = 0.0
    total_cost = 0.0

    for i, sample in enumerate(job.samples):
        if i == 0:
            dt_h = SAMPLE_INTERVAL / 3600
        else:
            dt_h = (sample.timestamp - job.samples[i - 1].timestamp) / 3600

        energy_wh = sample.total_attributed_w * dt_h
        total_energy_wh += energy_wh
        total_cost += (energy_wh / 1000) * sample.price_eur_per_kwh

    duration_s = job.samples[-1].timestamp - job.started_at + SAMPLE_INTERVAL
    energy_kwh = total_energy_wh / 1000
    avg_power_w = total_energy_wh / (duration_s / 3600) if duration_s > 0 else 0

    return {
        "energy_kwh": round(energy_kwh, 6),
        "cost_eur": round(total_cost, 6),
        "duration_s": round(duration_s, 1),
        "avg_power_w": round(avg_power_w, 1),
        "samples": len(job.samples),
    }


def sampling_loop():
    while True:
        try:
            shelly_w = monitor.read_shelly()
            gpu_powers = monitor.read_gpu_powers()
            gpu_utils = monitor.read_gpu_utilizations()

            monitor.update_baseline(shelly_w, gpu_utils)

            system_base = monitor.system_base
            total_gpu_w = sum(gpu_powers.values())
            overhead = max(0, shelly_w - system_base - total_gpu_w)

            price = price_tracker.current_price_eur_per_kwh()

            with lock:
                for job in active_jobs.values():
                    gpu_w = gpu_powers.get(job.gpu_index, 0)

                    if total_gpu_w > 0:
                        overhead_share = overhead * (gpu_w / total_gpu_w)
                    elif len(active_jobs) > 0:
                        overhead_share = overhead / len(active_jobs)
                    else:
                        overhead_share = 0

                    job.samples.append(JobSample(
                        timestamp=time.time(),
                        gpu_power_w=gpu_w,
                        overhead_share_w=round(overhead_share, 2),
                        total_attributed_w=round(gpu_w + overhead_share, 2),
                        price_eur_per_kwh=price,
                    ))

        except Exception:
            logger.exception("Sampling error")

        time.sleep(SAMPLE_INTERVAL)


@app.on_event("startup")
def startup():
    monitor.init_nvml()
    price_tracker.start()
    threading.Thread(target=sampling_loop, daemon=True).start()
    logger.info("Cost tracking service started (sample interval: %ds)", SAMPLE_INTERVAL)


@app.post("/job/start")
def job_start(req: JobStartRequest):
    job_id = uuid.uuid4().hex[:12]
    with lock:
        # Check no other job on same GPU
        for j in active_jobs.values():
            if j.gpu_index == req.gpu:
                raise HTTPException(409, f"GPU {req.gpu} already has active job {j.job_id}")

        active_jobs[job_id] = ActiveJob(
            job_id=job_id,
            gpu_index=req.gpu,
            pid=req.pid,
            client=req.client,
            label=req.label,
            started_at=time.time(),
        )

    logger.info("Job %s started on GPU %d (pid=%s)", job_id, req.gpu, req.pid)
    return {"job_id": job_id}


@app.post("/job/stop")
def job_stop(req: JobStopRequest):
    with lock:
        job = active_jobs.pop(req.job_id, None)

    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")

    result = compute_job_costs(job)
    logger.info("Job %s stopped: %s", req.job_id, result)
    return result


@app.get("/status")
def status():
    shelly_w = monitor.read_shelly()
    gpu_powers = monitor.read_gpu_powers()
    gpu_utils = monitor.read_gpu_utilizations()
    price = price_tracker.current_price_eur_per_kwh()

    with lock:
        jobs = {
            jid: {
                "gpu": j.gpu_index,
                "pid": j.pid,
                "client": j.client,
                "label": j.label,
                "started_at": datetime.fromtimestamp(j.started_at, tz=timezone.utc).isoformat(),
                "current_w": j.samples[-1].total_attributed_w if j.samples else 0,
                "gpu_w": j.samples[-1].gpu_power_w if j.samples else 0,
                "overhead_w": j.samples[-1].overhead_share_w if j.samples else 0,
                "samples": len(j.samples),
                "running": compute_job_costs(j),
            }
            for jid, j in active_jobs.items()
        }

    return {
        "shelly_total_w": shelly_w,
        "gpu_powers_w": gpu_powers,
        "gpu_utilizations_pct": gpu_utils,
        "system_base_w": monitor.system_base,
        "price_eur_per_kwh": price,
        "active_jobs": jobs,
    }


@app.get("/jobs")
def jobs_live():
    """Lightweight endpoint for UI polling — just active job power and cost."""
    with lock:
        return [
            {
                "job_id": jid,
                "gpu": j.gpu_index,
                "client": j.client,
                "label": j.label,
                "current_w": j.samples[-1].total_attributed_w if j.samples else 0,
                "energy_kwh": compute_job_costs(j)["energy_kwh"],
                "cost_eur": compute_job_costs(j)["cost_eur"],
                "duration_s": time.time() - j.started_at,
            }
            for jid, j in active_jobs.items()
        ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8377)
