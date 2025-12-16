"""
Logging and Monitoring Utilities for the Vision-Language-Action System
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os


class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors and specific format for different log levels"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SystemLogger:
    """Main logger for the VLA system"""

    def __init__(self, name: str = "VLA_System", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent adding multiple handlers if logger already exists
        if not self.logger.handlers:
            # Create console handler with custom formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(CustomFormatter())

            # Create file handler
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"vla_system_{timestamp}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)

            # Add handlers to logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


class PerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}

    def start_timer(self, operation_name: str):
        """Start timing for an operation"""
        import time
        self.start_times[operation_name] = time.time()
        self.logger.debug(f"Started timing for operation: {operation_name}")

    def end_timer(self, operation_name: str, log_result: bool = True) -> float:
        """End timing for an operation and return elapsed time"""
        import time
        if operation_name not in self.start_times:
            self.logger.warning(f"No start time recorded for operation: {operation_name}")
            return 0.0

        elapsed = time.time() - self.start_times[operation_name]
        if log_result:
            self.logger.info(f"Operation {operation_name} took {elapsed:.4f} seconds")

        # Store in metrics
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        self.metrics[operation_name].append(elapsed)

        # Clean up start time
        del self.start_times[operation_name]
        return elapsed

    def record_metric(self, metric_name: str, value: Any):
        """Record a specific metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        self.logger.debug(f"Recorded metric {metric_name}: {value}")

    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average value for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        values = [v for v in self.metrics[metric_name] if isinstance(v, (int, float))]
        if not values:
            return None

        avg = sum(values) / len(values)
        return avg

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded metrics"""
        stats = {}
        for metric_name, values in self.metrics.items():
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                stats[metric_name] = {
                    'count': len(numeric_values),
                    'average': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values)
                }
        return stats

    def save_metrics(self, filename: str = None):
        """Save metrics to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        self.logger.info(f"Performance metrics saved to {filename}")


class SafetyMonitor:
    """Monitor safety-related metrics and states"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.safety_violations = []
        self.heartbeat_received = True
        self.last_heartbeat_time = None
        self.emergency_stop_active = False

    def log_safety_violation(self, violation_type: str, details: str = ""):
        """Log a safety violation"""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'details': details
        }
        self.safety_violations.append(violation)
        self.logger.warning(f"Safety violation: {violation_type} - {details}")

    def trigger_emergency_stop(self, reason: str = ""):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

    def clear_emergency_stop(self):
        """Clear emergency stop"""
        self.emergency_stop_active = False
        self.logger.info("Emergency stop cleared")

    def update_heartbeat(self):
        """Update heartbeat status"""
        import time
        self.heartbeat_received = True
        self.last_heartbeat_time = time.time()
        self.logger.debug("Heartbeat received")

    def check_heartbeat_timeout(self, timeout_seconds: float = 3.0) -> bool:
        """Check if heartbeat has timed out"""
        import time
        if self.last_heartbeat_time is None:
            return True  # No heartbeat ever received

        if time.time() - self.last_heartbeat_time > timeout_seconds:
            self.heartbeat_received = False
            self.log_safety_violation("HEARTBEAT_TIMEOUT", f"Heartbeat timeout after {timeout_seconds}s")
            return True

        return False

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            'heartbeat_received': self.heartbeat_received,
            'emergency_stop_active': self.emergency_stop_active,
            'last_heartbeat_time': self.last_heartbeat_time,
            'safety_violations_count': len(self.safety_violations),
            'safety_violations': self.safety_violations[-10:]  # Last 10 violations
        }


# Global logger instance
def get_system_logger(name: str = "VLA_System") -> logging.Logger:
    """Get a configured system logger"""
    system_logger = SystemLogger(name)
    return system_logger.get_logger()


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_system_logger()

    # Test logging
    logger.info("Testing system logger")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test performance monitor
    perf_monitor = PerformanceMonitor(logger)
    perf_monitor.start_timer("test_operation")
    import time
    time.sleep(0.1)  # Simulate work
    elapsed = perf_monitor.end_timer("test_operation")
    print(f"Elapsed time: {elapsed:.4f}s")

    # Test safety monitor
    safety_monitor = SafetyMonitor(logger)
    safety_monitor.log_safety_violation("TEST_VIOLATION", "This is a test violation")
    print("Safety status:", safety_monitor.get_safety_status())