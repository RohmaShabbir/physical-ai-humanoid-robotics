"""
Performance Metrics Collection Framework for the Vision-Language-Action System
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import csv
import os
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be collected"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    SUCCESS_RATE = "success_rate"


@dataclass
class PerformanceMetric:
    """Data class for a single performance metric"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    source_component: str
    unit: str = ""
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['metric_type'] = self.metric_type.value
        result['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result


class MetricsCollector:
    """Main class for collecting performance metrics"""

    def __init__(self, output_dir: str = "metrics", logger=None):
        self.metrics: List[PerformanceMetric] = []
        self.output_dir = output_dir
        self.logger = logger
        self.lock = threading.Lock()
        self.session_start_time = time.time()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Callbacks to be called when metrics are collected
        self.callbacks: List[Callable[[PerformanceMetric], None]] = []

    def add_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add a callback to be called when metrics are collected"""
        self.callbacks.append(callback)

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        source_component: str,
        unit: str = "",
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        with self.lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=time.time(),
                source_component=source_component,
                unit=unit,
                additional_data=additional_data
            )
            self.metrics.append(metric)

            # Call all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in metrics callback: {e}")

    def record_latency(
        self,
        operation_name: str,
        latency_ms: float,
        source_component: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record a latency metric"""
        self.record_metric(
            name=f"{operation_name}_latency",
            value=latency_ms,
            metric_type=MetricType.LATENCY,
            source_component=source_component,
            unit="ms",
            additional_data=additional_data
        )

    def record_throughput(
        self,
        operation_name: str,
        throughput_value: float,
        source_component: str,
        unit: str = "items/sec",
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record a throughput metric"""
        self.record_metric(
            name=f"{operation_name}_throughput",
            value=throughput_value,
            metric_type=MetricType.THROUGHPUT,
            source_component=source_component,
            unit=unit,
            additional_data=additional_data
        )

    def record_accuracy(
        self,
        task_name: str,
        accuracy_value: float,
        source_component: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record an accuracy metric (0.0 to 1.0)"""
        self.record_metric(
            name=f"{task_name}_accuracy",
            value=accuracy_value,
            metric_type=MetricType.ACCURACY,
            source_component=source_component,
            unit="ratio",
            additional_data=additional_data
        )

    def record_resource_usage(
        self,
        resource_name: str,
        usage_value: float,
        source_component: str,
        unit: str = "%",
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record a resource usage metric"""
        self.record_metric(
            name=f"{resource_name}_usage",
            value=usage_value,
            metric_type=MetricType.RESOURCE_USAGE,
            source_component=source_component,
            unit=unit,
            additional_data=additional_data
        )

    def record_success_rate(
        self,
        operation_name: str,
        success_rate: float,
        source_component: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record a success rate metric (0.0 to 1.0)"""
        self.record_metric(
            name=f"{operation_name}_success_rate",
            value=success_rate,
            metric_type=MetricType.SUCCESS_RATE,
            source_component=source_component,
            unit="ratio",
            additional_data=additional_data
        )

    def get_metrics_by_type(self, metric_type: MetricType) -> List[PerformanceMetric]:
        """Get all metrics of a specific type"""
        with self.lock:
            return [m for m in self.metrics if m.metric_type == metric_type]

    def get_metrics_by_source(self, source_component: str) -> List[PerformanceMetric]:
        """Get all metrics from a specific source component"""
        with self.lock:
            return [m for m in self.metrics if m.source_component == source_component]

    def get_latest_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get the most recent metric with the given name"""
        with self.lock:
            for metric in reversed(self.metrics):
                if metric.name == name:
                    return metric
            return None

    def calculate_average(self, name: str) -> Optional[float]:
        """Calculate the average value for metrics with the given name"""
        with self.lock:
            matching_metrics = [m for m in self.metrics if m.name == name]
            if not matching_metrics:
                return None
            values = [m.value for m in matching_metrics]
            return sum(values) / len(values)

    def calculate_session_duration(self) -> float:
        """Calculate the duration of the current session in seconds"""
        return time.time() - self.session_start_time

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"metrics_{timestamp}.json")

        with self.lock:
            metrics_data = [metric.to_dict() for metric in self.metrics]
            session_info = {
                "session_start": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "session_duration": self.calculate_session_duration(),
                "total_metrics": len(self.metrics),
                "metrics": metrics_data
            }

        with open(filename, 'w') as f:
            json.dump(session_info, f, indent=2)

        if self.logger:
            self.logger.info(f"Metrics exported to {filename}")

        return filename

    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """Export metrics to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"metrics_{timestamp}.csv")

        with self.lock:
            with open(filename, 'w', newline='') as f:
                fieldnames = [
                    'name', 'value', 'metric_type', 'timestamp', 'source_component',
                    'unit', 'additional_data'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                for metric in self.metrics:
                    metric_dict = metric.to_dict()
                    # Convert additional_data to string for CSV
                    if metric_dict['additional_data']:
                        metric_dict['additional_data'] = json.dumps(metric_dict['additional_data'])
                    writer.writerow(metric_dict)

        if self.logger:
            self.logger.info(f"Metrics exported to {filename}")

        return filename

    def reset(self):
        """Reset all collected metrics"""
        with self.lock:
            self.metrics.clear()
            self.session_start_time = time.time()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics"""
        with self.lock:
            if not self.metrics:
                return {}

            # Group metrics by name
            metrics_by_name = {}
            for metric in self.metrics:
                if metric.name not in metrics_by_name:
                    metrics_by_name[metric.name] = []
                metrics_by_name[metric.name].append(metric)

            summary = {}
            for name, metrics_list in metrics_by_name.items():
                values = [m.value for m in metrics_list]
                summary[name] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }

            return summary


class MetricsTimer:
    """Context manager and decorator for timing operations and recording metrics"""

    def __init__(self, collector: MetricsCollector, metric_name: str, source_component: str):
        self.collector = collector
        self.metric_name = metric_name
        self.source_component = source_component
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.record_latency(
                operation_name=self.metric_name,
                latency_ms=duration_ms,
                source_component=self.source_component
            )

    def __call__(self, func):
        """Use as a decorator"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class ResourceMonitor:
    """Monitor system resources like CPU, memory, and VRAM"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system resources in a separate thread"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring system resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: float):
        """Internal monitoring loop"""
        import psutil
        try:
            while self.monitoring:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.collector.record_resource_usage(
                    resource_name="cpu",
                    usage_value=cpu_percent,
                    source_component="system_monitor"
                )

                # Memory usage
                memory_percent = psutil.virtual_memory().percent
                self.collector.record_resource_usage(
                    resource_name="memory",
                    usage_value=memory_percent,
                    source_component="system_monitor"
                )

                # Note: VRAM monitoring would require additional libraries like pynvml
                # For now, we'll simulate VRAM usage
                # In a real implementation, you would use pynvml to get GPU memory usage
                simulated_vram_usage = 45.0  # Simulated value
                self.collector.record_resource_usage(
                    resource_name="vram",
                    usage_value=simulated_vram_usage,
                    source_component="system_monitor",
                    unit="%"
                )

                time.sleep(interval)
        except Exception as e:
            if self.collector.logger:
                self.collector.logger.error(f"Error in resource monitoring: {e}")
            self.monitoring = False


# Global metrics collector instance
_default_collector: Optional[MetricsCollector] = None


def get_metrics_collector(output_dir: str = "metrics", logger=None) -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector(output_dir=output_dir, logger=logger)
    return _default_collector


# Example usage and testing
if __name__ == "__main__":
    import time
    import random

    # Create a metrics collector
    collector = MetricsCollector(output_dir="test_metrics")

    # Record some example metrics
    collector.record_latency("voice_processing", 150.5, "voice_pipeline")
    collector.record_throughput("vla_inference", 8.2, "vla_inference", "tokens/sec")
    collector.record_accuracy("object_detection", 0.92, "vla_inference")
    collector.record_resource_usage("gpu_memory", 65.3, "vla_inference", "%")
    collector.record_success_rate("grasp_action", 0.87, "skill_execution")

    # Use the timer context manager
    timer = MetricsTimer(collector, "test_operation", "test_component")
    with timer:
        time.sleep(0.1)  # Simulate some work

    # Use as a decorator
    @timer
    def example_function():
        time.sleep(0.05)  # Simulate more work
        return "result"

    result = example_function()

    # Print summary
    print("Summary Statistics:")
    summary = collector.get_summary_stats()
    for name, stats in summary.items():
        print(f"  {name}: avg={stats['average']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    # Export to files
    json_file = collector.export_to_json()
    csv_file = collector.export_to_csv()
    print(f"Metrics exported to {json_file} and {csv_file}")