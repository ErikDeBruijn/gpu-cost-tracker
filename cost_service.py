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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from epex_prices import EPEXPriceTracker
from power_monitor import PowerMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 5  # seconds
HISTORY_MAX = 17280  # 24 hours at 5s intervals


@dataclass
class SystemSample:
    timestamp: float
    shelly_w: float
    gpu_powers_w: dict[int, float]
    gpu_utils_pct: dict[int, int]
    system_base_w: float
    price_eur_per_kwh: float


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
  GET  /history     - Historic power samples (default: last 60 min, ?minutes=N)
  GET  /chart       - Live-updating power usage chart
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
history: deque[SystemSample] = deque(maxlen=HISTORY_MAX)
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

            history.append(SystemSample(
                timestamp=time.time(),
                shelly_w=shelly_w,
                gpu_powers_w=dict(gpu_powers),
                gpu_utils_pct=dict(gpu_utils),
                system_base_w=system_base,
                price_eur_per_kwh=price,
            ))

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


@app.get("/history")
def get_history(minutes: int = 60):
    """Return system power samples for the last N minutes."""
    cutoff = time.time() - (minutes * 60)
    # Downsample if too many points: target ~360 points max for smooth charting
    samples = [s for s in history if s.timestamp >= cutoff]
    step = max(1, len(samples) // 360)
    sampled = samples[::step]
    return [
        {
            "t": s.timestamp,
            "shelly": round(s.shelly_w, 1),
            "gpu": {str(k): round(v, 1) for k, v in s.gpu_powers_w.items()},
            "base": round(s.system_base_w, 1),
            "price": round(s.price_eur_per_kwh, 4),
        }
        for s in sampled
    ]


@app.get("/chart", response_class=HTMLResponse)
def chart():
    return CHART_HTML


CHART_HTML = """<!DOCTYPE html>
<html><head>
<title>GPU Cost Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
  h1 { margin: 0 0 10px; font-size: 1.4em; }
  .stats { display: flex; gap: 20px; margin-bottom: 15px; font-size: 0.95em; }
  .stat { background: #222; padding: 8px 14px; border-radius: 6px; }
  .stat .val { font-size: 1.3em; font-weight: bold; }
  .stat .label { color: #888; font-size: 0.85em; }
  canvas { background: #1a1a1a; border-radius: 8px; }
</style>
</head><body>
<h1>GPU Cost Tracker</h1>
<div class="stats">
  <div class="stat"><div class="val" id="total-w">--</div><div class="label">Total W</div></div>
  <div class="stat"><div class="val" id="gpu0-w">--</div><div class="label">GPU 0</div></div>
  <div class="stat"><div class="val" id="gpu1-w">--</div><div class="label">GPU 1</div></div>
  <div class="stat"><div class="val" id="base-w">--</div><div class="label">Baseline</div></div>
  <div class="stat"><div class="val" id="price">--</div><div class="label">EUR/kWh</div></div>
</div>
<canvas id="chart" height="100"></canvas>
<script>
const ctx = document.getElementById('chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    datasets: [
      { label: 'Total (Shelly)', borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', fill: true, pointRadius: 0, borderWidth: 1.5, data: [] },
      { label: 'GPU 0', borderColor: '#3b82f6', pointRadius: 0, borderWidth: 1.5, data: [] },
      { label: 'GPU 1', borderColor: '#10b981', pointRadius: 0, borderWidth: 1.5, data: [] },
      { label: 'Baseline', borderColor: '#666', borderDash: [5,5], pointRadius: 0, borderWidth: 1, data: [] },
    ]
  },
  options: {
    responsive: true,
    animation: false,
    interaction: { intersect: false, mode: 'index' },
    scales: {
      x: { type: 'time', time: { tooltipFormat: 'HH:mm:ss' }, ticks: { color: '#888' }, grid: { color: '#333' } },
      y: { title: { display: true, text: 'Watts', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#333' }, min: 0 }
    },
    plugins: { legend: { labels: { color: '#ccc' } } }
  }
});

async function update() {
  try {
    const [histRes, statusRes] = await Promise.all([
      fetch('/history?minutes=60'),
      fetch('/status')
    ]);
    const hist = await histRes.json();
    const status = await statusRes.json();

    chart.data.datasets[0].data = hist.map(s => ({ x: s.t * 1000, y: s.shelly }));
    chart.data.datasets[1].data = hist.map(s => ({ x: s.t * 1000, y: s.gpu['0'] || 0 }));
    chart.data.datasets[2].data = hist.map(s => ({ x: s.t * 1000, y: s.gpu['1'] || 0 }));
    chart.data.datasets[3].data = hist.map(s => ({ x: s.t * 1000, y: s.base }));
    chart.update();

    document.getElementById('total-w').textContent = status.shelly_total_w.toFixed(0) + 'W';
    document.getElementById('gpu0-w').textContent = (status.gpu_powers_w['0'] || 0).toFixed(0) + 'W';
    document.getElementById('gpu1-w').textContent = (status.gpu_powers_w['1'] || 0).toFixed(0) + 'W';
    document.getElementById('base-w').textContent = (status.system_base_w || 0).toFixed(0) + 'W';
    document.getElementById('price').textContent = status.price_eur_per_kwh.toFixed(3);
  } catch(e) { console.error(e); }
}

update();
setInterval(update, 5000);
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8377)
