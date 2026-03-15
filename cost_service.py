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
    gpu_temps_c: dict[int, int]
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



app = FastAPI(
    title="GPU Cost Tracker",
    description="Tracks real energy cost of GPU jobs using Shelly power monitoring, NVML, and EPEX spot prices.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
def root():
    return DASHBOARD_HTML
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
            gpu_temps = monitor.read_gpu_temperatures()

            monitor.update_baseline(shelly_w, gpu_utils)

            system_base = monitor.system_base or 0
            total_gpu_w = sum(gpu_powers.values())
            overhead = max(0, shelly_w - system_base - total_gpu_w)

            price = price_tracker.current_price_eur_per_kwh()

            history.append(SystemSample(
                timestamp=time.time(),
                shelly_w=shelly_w,
                gpu_powers_w=dict(gpu_powers),
                gpu_utils_pct=dict(gpu_utils),
                gpu_temps_c=dict(gpu_temps),
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


@app.get("/job/{job_id}")
def job_get(job_id: str):
    """Get current cost of a running job without stopping it."""
    with lock:
        job = active_jobs.get(job_id)

    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    result = compute_job_costs(job)
    result["gpu"] = job.gpu_index
    result["client"] = job.client
    result["label"] = job.label
    return result


@app.delete("/job/{job_id}")
def job_stop(job_id: str):
    """Stop tracking a job and return final cost."""
    with lock:
        job = active_jobs.pop(job_id, None)

    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    result = compute_job_costs(job)
    logger.info("Job %s stopped: %s", job_id, result)
    return result


@app.get("/status")
def status():
    shelly_w = monitor.read_shelly()
    gpu_powers = monitor.read_gpu_powers()
    gpu_utils = monitor.read_gpu_utilizations()
    gpu_temps = monitor.read_gpu_temperatures()
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
        "gpu_temperatures_c": gpu_temps,
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
    result = []
    for s in sampled:
        base = round(s.system_base_w or 0, 1)
        gpu0 = round(s.gpu_powers_w.get(0, 0), 1)
        gpu1 = round(s.gpu_powers_w.get(1, 0), 1)
        rest = round(max(0, s.shelly_w - base - gpu0 - gpu1), 1)
        result.append({
            "t": s.timestamp,
            "shelly": round(s.shelly_w, 1),
            "base": base,
            "gpu0": gpu0,
            "gpu1": gpu1,
            "rest": rest,
            "util0": s.gpu_utils_pct.get(0, 0),
            "util1": s.gpu_utils_pct.get(1, 0),
            "temp0": s.gpu_temps_c.get(0, 0),
            "temp1": s.gpu_temps_c.get(1, 0),
        })
    return result


@app.post("/refresh-prices")
def refresh_prices():
    """Trigger an EPEX price refresh. Called by cron at 13:15 daily."""
    price_tracker.refresh()
    return {"status": "ok", "price_eur_per_kwh": price_tracker.current_price_eur_per_kwh()}


@app.get("/chart", response_class=HTMLResponse)
def chart():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<title>GPU Cost Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto; }
  h1 { margin: 0 0 10px; font-size: 1.4em; }
  .stats { display: flex; gap: 12px; margin-bottom: 15px; font-size: 0.95em; flex-wrap: wrap; }
  .stat { background: #222; padding: 8px 14px; border-radius: 6px; min-width: 90px; }
  .stat .val { font-size: 1.3em; font-weight: bold; }
  .stat .label { color: #888; font-size: 0.85em; }
  canvas { background: #1a1a1a; border-radius: 8px; }

  h2 { font-size: 1.1em; margin: 20px 0 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  th { text-align: left; color: #888; font-weight: 500; padding: 6px 10px; border-bottom: 1px solid #333; }
  td { padding: 6px 10px; border-bottom: 1px solid #222; }
  .no-jobs { color: #666; font-style: italic; padding: 10px; }

  details { margin-top: 20px; background: #1a1a1a; border-radius: 8px; }
  summary { cursor: pointer; padding: 12px 16px; color: #888; font-size: 0.95em; user-select: none; }
  summary:hover { color: #ccc; }
  .api-docs { padding: 0 16px 16px; }
  .api-docs pre { background: #222; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; line-height: 1.5; }
  .api-docs code { color: #f59e0b; }
</style>
</head><body>
<h1>GPU Cost Tracker <a href="https://github.com/ErikDeBruijn/gpu-cost-tracker" style="font-size:0.6em;color:#888;text-decoration:none;margin-left:8px">GitHub</a></h1>

<div class="stats">
  <div class="stat"><div class="val" id="total-w">--</div><div class="label">Total W</div></div>
  <div class="stat"><div class="val" id="gpu0-w">--</div><div class="label">GPU 0</div></div>
  <div class="stat"><div class="val" id="gpu0-info">--</div><div class="label">Util / Temp</div></div>
  <div class="stat"><div class="val" id="gpu1-w">--</div><div class="label">GPU 1</div></div>
  <div class="stat"><div class="val" id="gpu1-info">--</div><div class="label">Util / Temp</div></div>
  <div class="stat"><div class="val" id="base-w">--</div><div class="label">Baseline</div></div>
  <div class="stat"><div class="val" id="price">--</div><div class="label">EUR/kWh</div></div>
</div>

<canvas id="chart" height="100"></canvas>

<h2>Active Jobs</h2>
<div id="jobs-container"><div class="no-jobs">No active jobs</div></div>

<details>
  <summary>API Documentation</summary>
  <div class="api-docs">
  <pre>
<code>POST   /job/start</code>      Start tracking a job
       Body: {"gpu": 0, "client": "myapp", "label": "run_42"}
       Returns: {"job_id": "abc123"}

<code>GET    /job/{job_id}</code>  Get current cost of a running job
<code>DELETE /job/{job_id}</code>  Stop tracking, return final cost
       Returns: {"energy_kwh": 0.12, "cost_eur": 0.03,
                 "duration_s": 300, "avg_power_w": 280}

<code>GET    /jobs</code>          Live power + cost for all active jobs
<code>GET    /status</code>        Full system state (power, util%, temp, price, jobs)
<code>GET    /history</code>       Power samples with util% and temp (default: ?minutes=60)
<code>POST   /refresh-prices</code> Trigger EPEX price refresh
<code>GET    /openapi.json</code>  OpenAPI schema
<code>GET    /docs</code>          Swagger UI

<strong>Status response includes:</strong>
  shelly_total_w, gpu_powers_w, gpu_utilizations_pct,
  gpu_temperatures_c, system_base_w, price_eur_per_kwh, active_jobs

<strong>Attribution model:</strong>
  System baseline (idle power) is excluded.
  GPU power attributed directly per GPU via NVML.
  Overhead (CPU, fans) split proportionally by GPU power.
  Cost = energy * (EPEX spot + 0.125 taxes + 0.02 purchasing)
  </pre>
  </div>
</details>

<script>
const ctx = document.getElementById('chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    datasets: [
      { label: 'Baseline', borderColor: '#555', backgroundColor: 'rgba(85,85,85,0.6)', fill: true, pointRadius: 0, borderWidth: 0.5, data: [] },
      { label: 'GPU 0', borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.6)', fill: true, pointRadius: 0, borderWidth: 0.5, data: [] },
      { label: 'GPU 1', borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.6)', fill: true, pointRadius: 0, borderWidth: 0.5, data: [] },
      { label: 'Other', borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.6)', fill: true, pointRadius: 0, borderWidth: 0.5, data: [] },
    ]
  },
  options: {
    responsive: true,
    animation: false,
    interaction: { intersect: false, mode: 'index' },
    scales: {
      x: { type: 'time', time: { tooltipFormat: 'HH:mm:ss' }, ticks: { color: '#888' }, grid: { color: '#333' } },
      y: { stacked: true, title: { display: true, text: 'Watts', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#333' }, min: 0 }
    },
    plugins: { legend: { labels: { color: '#ccc' } } }
  }
});

function formatDuration(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'm';
  return (s/3600).toFixed(1) + 'h';
}

async function update() {
  try {
    const [histRes, statusRes, jobsRes] = await Promise.all([
      fetch('/history?minutes=60'),
      fetch('/status'),
      fetch('/jobs'),
    ]);
    const hist = await histRes.json();
    const status = await statusRes.json();
    const jobs = await jobsRes.json();

    chart.data.datasets[0].data = hist.map(s => ({ x: s.t * 1000, y: s.base }));
    chart.data.datasets[1].data = hist.map(s => ({ x: s.t * 1000, y: s.gpu0 }));
    chart.data.datasets[2].data = hist.map(s => ({ x: s.t * 1000, y: s.gpu1 }));
    chart.data.datasets[3].data = hist.map(s => ({ x: s.t * 1000, y: s.rest }));
    chart.update();

    document.getElementById('total-w').textContent = status.shelly_total_w.toFixed(0) + 'W';
    document.getElementById('gpu0-w').textContent = (status.gpu_powers_w['0'] || 0).toFixed(0) + 'W';
    document.getElementById('gpu0-info').textContent = (status.gpu_utilizations_pct['0'] || 0) + '% / ' + (status.gpu_temperatures_c['0'] || 0) + '\u00B0C';
    document.getElementById('gpu1-w').textContent = (status.gpu_powers_w['1'] || 0).toFixed(0) + 'W';
    document.getElementById('gpu1-info').textContent = (status.gpu_utilizations_pct['1'] || 0) + '% / ' + (status.gpu_temperatures_c['1'] || 0) + '\u00B0C';
    document.getElementById('base-w').textContent = (status.system_base_w || 0).toFixed(0) + 'W';
    document.getElementById('price').textContent = status.price_eur_per_kwh.toFixed(3);

    const container = document.getElementById('jobs-container');
    if (jobs.length === 0) {
      container.innerHTML = '<div class="no-jobs">No active jobs</div>';
    } else {
      container.innerHTML = `<table>
        <tr><th>GPU</th><th>Client</th><th>Label</th><th>Power</th><th>Duration</th><th>Energy</th><th>Cost</th></tr>
        ${jobs.map(j => `<tr>
          <td>${j.gpu}</td>
          <td>${j.client || '-'}</td>
          <td>${j.label || '-'}</td>
          <td>${j.current_w.toFixed(0)}W</td>
          <td>${formatDuration(j.duration_s)}</td>
          <td>${j.energy_kwh.toFixed(4)} kWh</td>
          <td>&euro;${j.cost_eur.toFixed(4)}</td>
        </tr>`).join('')}
      </table>`;
    }
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
