"""
Latency Measurement Test for the Vision-Language-Action System
Measures end-to-end latency from speech recognition to action start < 2.0 seconds
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logging_utils import get_system_logger
from src.vla_integration.voice_to_vla_adapter import VoiceToVLAAdapter, VoiceToVLARequest
from src.voice_pipeline.voice_service import VoiceCommandService
from src.vla_inference.inference_service import VLAInferenceService
from src.skill_library.skill_chain import SkillChainExecutor


@dataclass
class LatencyMeasurement:
    """
    Data class for latency measurement result
    """
    test_name: str
    total_latency: float  # Total time from speech to action start
    component_latencies: Dict[str, float]  # Latency of each component
    success: bool  # Whether latency meets requirements
    timestamp: float
    details: Dict[str, Any]


class LatencyTestSuite:
    """
    Test suite for measuring end-to-end latency in the VLA system
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the latency test suite

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("LatencyTestSuite")
        self.config = self._load_config(config_path)

        # Initialize components
        self.voice_service = VoiceCommandService()
        self.vla_service = VLAInferenceService()
        self.skill_executor = SkillChainExecutor()
        self.voice_to_vla_adapter = VoiceToVLAAdapter()

        # Latency requirements
        self.max_latency = self.config.get("max_latency", 2.0)  # 2.0 seconds
        self.target_latency = self.config.get("target_latency", 1.5)  # 1.5 seconds target

        # Test parameters
        self.test_commands = [
            "pick up the red cup",
            "go to the kitchen sink",
            "place the bottle on the table",
            "bring the red cup to the kitchen sink"
        ]

        self.logger.info(f"Latency Test Suite initialized with max latency: {self.max_latency}s")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the latency test suite
        """
        default_config = {
            "max_latency": 2.0,  # seconds
            "target_latency": 1.5,  # seconds
            "num_test_iterations": 5,
            "warmup_iterations": 2,
            "enable_detailed_measurement": True,
            "test_objects": ["red cup", "blue bottle", "green book"],
            "test_locations": ["kitchen sink", "table", "counter"]
        }

        if config_path:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def measure_end_to_end_latency(self, command: str) -> LatencyMeasurement:
        """
        Measure end-to-end latency for a single command

        Args:
            command: The voice command to test

        Returns:
            Latency measurement result
        """
        start_time = time.time()
        component_latencies = {}

        try:
            self.logger.info(f"Measuring latency for command: '{command}'")

            # Step 1: Voice processing latency
            voice_start = time.time()
            voice_response = self.voice_service.process_voice_command_text(
                command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "robot_capabilities": ["grasp", "navigate", "place"]
                }
            )
            voice_latency = time.time() - voice_start
            component_latencies["voice_processing"] = voice_latency

            if not voice_response.success:
                return LatencyMeasurement(
                    test_name=f"Latency test for '{command}'",
                    total_latency=time.time() - start_time,
                    component_latencies=component_latencies,
                    success=False,
                    timestamp=start_time,
                    details={"error": voice_response.error_message}
                )

            # Step 2: VLA inference latency
            vla_start = time.time()
            vla_request = VLAInferenceRequest(
                image_data=None,  # Will use mock in testing
                text_instruction=command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "target_object": voice_response.command_structure.get("target_object", "object"),
                    "target_location": voice_response.command_structure.get("target_location", "location")
                }
            )
            vla_response = self.vla_service.execute(vla_request)
            vla_latency = time.time() - vla_start
            component_latencies["vla_inference"] = vla_latency

            if not vla_response.success:
                return LatencyMeasurement(
                    test_name=f"Latency test for '{command}'",
                    total_latency=time.time() - start_time,
                    component_latencies=component_latencies,
                    success=False,
                    timestamp=start_time,
                    details={"error": vla_response.error_message}
                )

            # Step 3: Skill preparation latency (getting ready to execute)
            skill_prep_start = time.time()
            # In a real system, this would be the time to prepare for action execution
            # For testing, we'll simulate a small delay
            time.sleep(0.05)  # Simulate skill preparation
            skill_prep_latency = time.time() - skill_prep_start
            component_latencies["skill_preparation"] = skill_prep_latency

            # Calculate total latency (speech to action start)
            total_latency = voice_latency + vla_latency + skill_prep_latency

            # Check if latency meets requirements
            success = total_latency <= self.max_latency

            return LatencyMeasurement(
                test_name=f"Latency test for '{command}'",
                total_latency=total_latency,
                component_latencies=component_latencies,
                success=success,
                timestamp=start_time,
                details={
                    "command": command,
                    "voice_latency": voice_latency,
                    "vla_latency": vla_latency,
                    "skill_prep_latency": skill_prep_latency,
                    "max_allowed_latency": self.max_latency,
                    "meets_requirement": success
                }
            )

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error measuring latency for '{command}': {e}")
            return LatencyMeasurement(
                test_name=f"Latency test for '{command}'",
                total_latency=total_time,
                component_latencies=component_latencies,
                success=False,
                timestamp=start_time,
                details={"error": str(e)}
            )

    def measure_full_pipeline_latency(self, command: str) -> LatencyMeasurement:
        """
        Measure latency through the full integrated pipeline

        Args:
            command: The voice command to test

        Returns:
            Latency measurement result
        """
        start_time = time.time()

        try:
            self.logger.info(f"Measuring full pipeline latency for command: '{command}'")

            # Measure through the integrated adapter
            adapter_start = time.time()
            request = VoiceToVLARequest(
                text_command=command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "available_locations": self.config.get("test_locations", []),
                    "robot_capabilities": ["grasp", "navigate", "place"]
                }
            )
            response = self.voice_to_vla_adapter.process_voice_command(request)
            adapter_latency = time.time() - adapter_start

            # For this measurement, we consider the time until the first skill is ready to execute
            # This represents the time from speech input to action start preparation
            total_latency = adapter_latency

            success = total_latency <= self.max_latency

            return LatencyMeasurement(
                test_name=f"Full pipeline latency test for '{command}'",
                total_latency=total_latency,
                component_latencies={"full_pipeline": adapter_latency},
                success=success,
                timestamp=start_time,
                details={
                    "command": command,
                    "full_pipeline_latency": adapter_latency,
                    "max_allowed_latency": self.max_latency,
                    "meets_requirement": success,
                    "adapter_response_success": response.success
                }
            )

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error measuring full pipeline latency for '{command}': {e}")
            return LatencyMeasurement(
                test_name=f"Full pipeline latency test for '{command}'",
                total_latency=total_time,
                component_latencies={},
                success=False,
                timestamp=start_time,
                details={"error": str(e)}
            )

    def run_latency_benchmark(self) -> List[LatencyMeasurement]:
        """
        Run comprehensive latency benchmark with multiple commands and iterations

        Returns:
            List of latency measurements
        """
        results = []
        num_iterations = self.config.get("num_test_iterations", 5)
        warmup_iterations = self.config.get("warmup_iterations", 2)

        self.logger.info(f"Running latency benchmark with {num_iterations} iterations")

        # Warmup runs
        self.logger.info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            for command in self.test_commands:
                self.measure_end_to_end_latency(command)
                time.sleep(0.1)  # Small delay between tests

        # Actual benchmark runs
        self.logger.info("Running actual benchmark iterations...")
        for iteration in range(num_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}")
            for command in self.test_commands:
                result = self.measure_end_to_end_latency(command)
                results.append(result)
                time.sleep(0.1)  # Small delay between tests

        return results

    def run_full_pipeline_benchmark(self) -> List[LatencyMeasurement]:
        """
        Run latency benchmark through the full integrated pipeline

        Returns:
            List of latency measurements
        """
        results = []
        num_iterations = self.config.get("num_test_iterations", 5)
        warmup_iterations = self.config.get("warmup_iterations", 2)

        self.logger.info(f"Running full pipeline benchmark with {num_iterations} iterations")

        # Warmup runs
        self.logger.info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            for command in self.test_commands:
                self.measure_full_pipeline_latency(command)
                time.sleep(0.1)

        # Actual benchmark runs
        self.logger.info("Running actual full pipeline benchmark iterations...")
        for iteration in range(num_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}")
            for command in self.test_commands:
                result = self.measure_full_pipeline_latency(command)
                results.append(result)
                time.sleep(0.1)

        return results

    def analyze_results(self, results: List[LatencyMeasurement]) -> Dict[str, Any]:
        """
        Analyze latency test results

        Args:
            results: List of latency measurements

        Returns:
            Analysis of the results
        """
        if not results:
            return {"error": "No results to analyze"}

        latencies = [r.total_latency for r in results]
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]

        # Calculate statistics
        avg_latency = np.mean(latencies) if latencies else 0
        min_latency = np.min(latencies) if latencies else 0
        max_latency = np.max(latencies) if latencies else 0
        std_latency = np.std(latencies) if len(latencies) > 1 else 0

        success_rate = len(successful_tests) / len(results) if results else 0

        # Analyze by command
        command_analysis = {}
        for command in self.test_commands:
            command_results = [r for r in results if command in r.test_name]
            if command_results:
                cmd_latencies = [r.total_latency for r in command_results]
                command_analysis[command] = {
                    "avg_latency": np.mean(cmd_latencies),
                    "min_latency": np.min(cmd_latencies),
                    "max_latency": np.max(cmd_latencies),
                    "success_rate": len([r for r in command_results if r.success]) / len(command_results)
                }

        analysis = {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": success_rate,
            "latency_stats": {
                "average": float(avg_latency),
                "minimum": float(min_latency),
                "maximum": float(max_latency),
                "std_deviation": float(std_latency),
                "max_allowed": self.max_latency
            },
            "by_command": command_analysis,
            "meets_requirements": success_rate >= 0.95,  # Require 95% success rate
            "summary": f"Success rate: {success_rate:.1%} ({len(successful_tests)}/{len(results)} tests under {self.max_latency}s)"
        }

        return analysis

    def generate_performance_report(self, results: List[LatencyMeasurement]) -> str:
        """
        Generate a performance report from the test results

        Args:
            results: List of latency measurements

        Returns:
            Formatted performance report
        """
        analysis = self.analyze_results(results)

        report = []
        report.append("=" * 60)
        report.append("VLA System Latency Performance Report")
        report.append("=" * 60)
        report.append(f"Max Allowed Latency: {self.max_latency}s")
        report.append(f"Target Latency: {self.target_latency}s")
        report.append(f"Total Tests Run: {analysis['total_tests']}")
        report.append(f"Success Rate: {analysis['success_rate']:.1%}")
        report.append("")

        # Latency statistics
        stats = analysis['latency_stats']
        report.append("Latency Statistics:")
        report.append(f"  Average: {stats['average']:.3f}s")
        report.append(f"  Minimum: {stats['minimum']:.3f}s")
        report.append(f"  Maximum: {stats['maximum']:.3f}s")
        report.append(f"  Std Dev: {stats['std_deviation']:.3f}s")
        report.append("")

        # By command analysis
        report.append("Performance by Command:")
        for cmd, cmd_stats in analysis['by_command'].items():
            report.append(f"  {cmd}:")
            report.append(f"    Avg: {cmd_stats['avg_latency']:.3f}s")
            report.append(f"    Success Rate: {cmd_stats['success_rate']:.1%}")
        report.append("")

        # Requirements check
        meets_req = analysis['meets_requirements']
        report.append(f"Requirements Check: {'PASS' if meets_req else 'FAIL'}")
        report.append(f"  > 95% tests under {self.max_latency}s: {'PASS' if meets_req else 'FAIL'}")
        report.append(f"  > Max latency requirement ({self.max_latency}s): {'PASS' if stats['maximum'] <= self.max_latency else 'FAIL'}")
        report.append("=" * 60)

        return "\n".join(report)

    def run_complete_latency_test(self) -> Dict[str, Any]:
        """
        Run the complete latency test and return comprehensive results

        Returns:
            Dictionary with complete test results
        """
        self.logger.info("Starting complete latency test...")

        # Run component-level benchmark
        component_results = self.run_latency_benchmark()

        # Run full pipeline benchmark
        pipeline_results = self.run_full_pipeline_benchmark()

        # Analyze results
        component_analysis = self.analyze_results(component_results)
        pipeline_analysis = self.analyze_results(pipeline_results)

        # Generate reports
        component_report = self.generate_performance_report(component_results)
        pipeline_report = self.generate_performance_report(pipeline_results)

        complete_results = {
            "component_level": {
                "results": component_results,
                "analysis": component_analysis,
                "report": component_report
            },
            "full_pipeline": {
                "results": pipeline_results,
                "analysis": pipeline_analysis,
                "report": pipeline_report
            },
            "overall_success": (
                component_analysis['meets_requirements'] and
                pipeline_analysis['meets_requirements']
            ),
            "summary": {
                "component_success_rate": component_analysis['success_rate'],
                "pipeline_success_rate": pipeline_analysis['success_rate'],
                "overall_meets_requirements": (
                    component_analysis['meets_requirements'] and
                    pipeline_analysis['meets_requirements']
                )
            }
        }

        # Log summary
        self.logger.info(f"Component-level success rate: {component_analysis['success_rate']:.1%}")
        self.logger.info(f"Full pipeline success rate: {pipeline_analysis['success_rate']:.1%}")
        self.logger.info(f"Overall requirements met: {complete_results['overall_success']}")

        return complete_results


def run_latency_tests():
    """
    Run the latency tests and return results
    """
    logger = get_system_logger("LatencyTestRunner")
    logger.info("Starting VLA System Latency Tests...")

    # Create the test suite
    test_suite = LatencyTestSuite()

    # Run the complete latency test
    results = test_suite.run_complete_latency_test()

    # Print reports
    print("\n" + results["component_level"]["report"])
    print("\n" + results["full_pipeline"]["report"])

    # Print summary
    summary = results["summary"]
    print(f"\nTest Summary:")
    print(f"  Component-level success rate: {summary['component_success_rate']:.1%}")
    print(f"  Full pipeline success rate: {summary['pipeline_success_rate']:.1%}")
    print(f"  Overall meets requirements: {summary['overall_meets_requirements']}")

    return results


if __name__ == "__main__":
    results = run_latency_tests()
    print(f"\nLatency tests completed.")
    print(f"Overall success: {'PASS' if results['overall_success'] else 'FAIL'}")