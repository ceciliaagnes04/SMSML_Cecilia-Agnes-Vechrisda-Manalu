from prometheus_client import start_http_server, Summary, Counter, Histogram
import time
import random

INFERENCE_TIME = Summary('churn_inference_processing_seconds', 'Waktu proses inferensi')
INFERENCE_COUNT = Counter('churn_inference_total', 'Total inferensi churn')
INFERENCE_ERROR_COUNT = Counter('churn_inference_error_total', 'Total inferensi churn yang gagal')
INFERENCE_STATUS = Counter('churn_inference_status', 'Status kode inferensi', ['status_code'])
INFERENCE_ERROR_STATUS = Counter('churn_inference_error_status', 'Status error inferensi', ['error_type'])
INFERENCE_LATENCY = Histogram('churn_inference_latency_seconds', 'Histogram waktu inferensi')

@INFERENCE_TIME.time()
@INFERENCE_LATENCY.time()
def process_inference(success=True):
    time.sleep(random.uniform(0.01, 0.1))
    INFERENCE_COUNT.inc()
    if success:
        INFERENCE_STATUS.labels(status_code="200").inc()
    else:
        INFERENCE_ERROR_COUNT.inc()
        INFERENCE_STATUS.labels(status_code="500").inc()
        INFERENCE_ERROR_STATUS.labels(error_type="internal_error").inc()

if __name__ == '__main__':
    # Inisialisasi label agar metrik selalu muncul di Prometheus
    INFERENCE_STATUS.labels(status_code="200")
    INFERENCE_STATUS.labels(status_code="500")
    INFERENCE_ERROR_STATUS.labels(error_type="internal_error")
    start_http_server(8001, addr="0.0.0.0")
    print("Prometheus exporter berjalan di port 8001")
    while True:
        # Simulasi error 1 dari 10
        process_inference(success=random.random() > 0.1)
        time.sleep(1)