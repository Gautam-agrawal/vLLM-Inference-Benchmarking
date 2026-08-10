"""
Prometheus + Grafana Monitoring Setup for vLLM (Local)

Downloads and launches Prometheus (scraping vLLM's /metrics endpoint) and
Grafana locally. Requires vLLM's server to already be running
(see vllm_benchmark.py).

Once running, access the dashboards directly at:
    Prometheus: http://localhost:9090
    Grafana:    http://localhost:3000  (default login: admin / admin)
"""

import fcntl
import os
import subprocess

PROMETHEUS_VERSION = "2.53.0"
GRAFANA_VERSION = "11.1.0"
VLLM_PORT = 8000
GRAFANA_PORT = 3000

PROMETHEUS_DIR = f"prometheus-{PROMETHEUS_VERSION}.linux-amd64"
GRAFANA_DIR = f"grafana-v{GRAFANA_VERSION}"


def check_vllm_metrics(port: int = VLLM_PORT) -> None:
    import requests
    r = requests.get(f"http://localhost:{port}/metrics")
    print(r.text[:1000])


def download_prometheus(version: str = PROMETHEUS_VERSION) -> None:
    url = (
        f"https://github.com/prometheus/prometheus/releases/download/"
        f"v{version}/prometheus-{version}.linux-amd64.tar.gz"
    )
    subprocess.run(["wget", "-q", url], check=True)
    subprocess.run(
        ["tar", "xzf", f"prometheus-{version}.linux-amd64.tar.gz"], check=True
    )


def write_prometheus_config(vllm_port: int = VLLM_PORT, prometheus_dir: str = PROMETHEUS_DIR) -> None:
    config = f"""
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:{vllm_port}']
"""
    with open(os.path.join(prometheus_dir, "prometheus.yml"), "w") as f:
        f.write(config)


def start_prometheus(prometheus_dir: str = PROMETHEUS_DIR) -> subprocess.Popen:
    return subprocess.Popen(
        ["./prometheus", "--config.file=prometheus.yml"],
        cwd=prometheus_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def download_grafana(version: str = GRAFANA_VERSION) -> None:
    url = f"https://dl.grafana.com/oss/release/grafana-{version}.linux-amd64.tar.gz"
    subprocess.run(["wget", "-q", url], check=True)
    subprocess.run(["tar", "xzf", f"grafana-{version}.linux-amd64.tar.gz"], check=True)


def start_grafana(grafana_dir: str = GRAFANA_DIR) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["./bin/grafana-server"],
        cwd=grafana_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Make stdout non-blocking so we can peek at logs without hanging
    fd = proc.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    return proc


if __name__ == "__main__":
    check_vllm_metrics()

    download_prometheus()
    write_prometheus_config()
    prom_proc = start_prometheus()

    download_grafana()
    grafana_proc = start_grafana()

    print("\nMonitoring stack is running locally:")
    print(f"  Prometheus: http://localhost:9090")
    print(f"  Grafana:    http://localhost:{GRAFANA_PORT}  (default login: admin / admin)")
    print("\nPress Ctrl+C to stop.")
