"""
Task Completion Time Verification for the Vision-Language-Action System
Verifies that tasks complete within 3 minutes in simulation (success criteria SC-001)
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
from src.skill_library.skill_chain import SkillChainExecutor, SkillChain
from src.ros_interfaces.message_converters import RobotSkill


@dataclass
class TaskCompletionResult:
    """
    Data class for task completion result
    """
    test_name: str
    task_name: str
    success: bool
    total_time: float  # Total time for task completion
    skill_execution_times: List[float]  # Time for each skill
    max_allowed_time: float  # Maximum allowed time (3 minutes = 180 seconds)
    timestamp: float
    details: Dict[str, Any]


class TaskCompletionTestSuite:
    """
    Test suite for verifying task completion within time limits
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the task completion test suite

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("TaskCompletionTestSuite")
        self.config = self._load_config(config_path)

        # Initialize components
        self.voice_to_vla_adapter = VoiceToVLAAdapter()
        self.skill_executor = SkillChainExecutor()

        # Time requirements
        self.max_task_time = self.config.get("max_task_time", 180.0)  # 3 minutes = 180 seconds
        self.target_task_time = self.config.get("target_task_time", 120.0)  # 2 minutes target

        # Define test tasks that should complete within 3 minutes
        self.test_tasks = [
            {
                "name": "bring_red_cup",
                "command": "Bring the red cup from the table to the kitchen sink",
                "description": "Navigate to table, grasp red cup, navigate to sink, place cup"
            },
            {
                "name": "tidy_apartment_simple",
                "command": "Pick up the blue bottle and place it on the counter",
                "description": "Grasp blue bottle, navigate to counter, place bottle"
            },
            {
                "name": "move_book",
                "command": "Take the green book from the desk to the living room",
                "description": "Navigate to desk, grasp book, navigate to living room, place book"
            }
        ]

        self.logger.info(f"Task Completion Test Suite initialized with max time: {self.max_task_time}s")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the task completion test suite
        """
        default_config = {
            "max_task_time": 180.0,  # 3 minutes in seconds
            "target_task_time": 120.0,  # 2 minutes target
            "num_test_iterations": 3,
            "warmup_iterations": 1,
            "enable_detailed_timing": True,
            "test_objects": ["red cup", "blue bottle", "green book", "white plate"],
            "test_locations": ["kitchen sink", "table", "counter", "desk", "living room"],
            "robot_capabilities": ["grasp", "navigate", "place"]
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

    def simulate_task_completion(self, task_name: str, command: str) -> TaskCompletionResult:
        """
        Simulate task completion and measure the time

        Args:
            task_name: Name of the task
            command: Voice command for the task

        Returns:
            Task completion result
        """
        start_time = time.time()
        skill_times = []

        try:
            self.logger.info(f"Simulating task completion for: '{task_name}' - '{command}'")

            # Create a request for the voice-to-VLA adapter
            request = VoiceToVLARequest(
                text_command=command,
                context={
                    "robot_state": "idle",
                    "environment_state": "apartment",
                    "available_objects": self.config.get("test_objects", []),
                    "available_locations": self.config.get("test_locations", []),
                    "robot_capabilities": self.config.get("robot_capabilities", [])
                }
            )

            # Process through the full pipeline
            response = self.voice_to_vla_adapter.process_voice_command(request)

            if not response.success:
                total_time = time.time() - start_time
                return TaskCompletionResult(
                    test_name=f"Task completion test for '{task_name}'",
                    task_name=task_name,
                    success=False,
                    total_time=total_time,
                    skill_execution_times=skill_times,
                    max_allowed_time=self.max_task_time,
                    timestamp=start_time,
                    details={
                        "command": command,
                        "error": response.error_message,
                        "voice_processing_time": getattr(response, 'total_processing_time', 0)
                    }
                )

            # If we have skill execution results, extract timing information
            if response.skill_execution_result:
                execution_results = response.skill_execution_result.get("skill_chain_execution", {}).get("results", [])
                for result in execution_results:
                    if "execution_time" in result:
                        skill_times.append(result["execution_time"])

            total_time = time.time() - start_time

            # Check if task completed within time limit
            success = total_time <= self.max_task_time

            return TaskCompletionResult(
                test_name=f"Task completion test for '{task_name}'",
                task_name=task_name,
                success=success,
                total_time=total_time,
                skill_execution_times=skill_times,
                max_allowed_time=self.max_task_time,
                timestamp=start_time,
                details={
                    "command": command,
                    "voice_processing_time": response.total_processing_time,
                    "skill_execution_count": len(skill_times),
                    "average_skill_time": np.mean(skill_times) if skill_times else 0,
                    "max_allowed_time": self.max_task_time,
                    "meets_time_requirement": success
                }
            )

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error simulating task completion for '{task_name}': {e}")
            return TaskCompletionResult(
                test_name=f"Task completion test for '{task_name}'",
                task_name=task_name,
                success=False,
                total_time=total_time,
                skill_execution_times=skill_times,
                max_allowed_time=self.max_task_time,
                timestamp=start_time,
                details={"error": str(e)}
            )

    def simulate_complex_task(self, task_name: str, skills: List[RobotSkill]) -> TaskCompletionResult:
        """
        Simulate completion of a complex task with multiple skills

        Args:
            task_name: Name of the task
            skills: List of RobotSkill objects for the task

        Returns:
            Task completion result
        """
        start_time = time.time()
        skill_times = []

        try:
            self.logger.info(f"Simulating complex task completion for: '{task_name}' with {len(skills)} skills")

            # Create a skill chain
            skill_chain = SkillChain(
                skills=skills,
                execution_order=list(range(len(skills))),
                dependencies={},
                context={"task_name": task_name, "simulation": True}
            )

            # Execute the skill chain
            chain_start = time.time()
            execution_success, execution_message, execution_results = self.skill_executor.execute_chain(skill_chain)
            chain_time = time.time() - chain_start

            # Extract individual skill execution times
            for result in execution_results:
                skill_time = result.end_time - result.start_time
                skill_times.append(skill_time)

            total_time = time.time() - start_time

            # Check if task completed within time limit
            success = total_time <= self.max_task_time

            return TaskCompletionResult(
                test_name=f"Complex task completion test for '{task_name}'",
                task_name=task_name,
                success=success,
                total_time=total_time,
                skill_execution_times=skill_times,
                max_allowed_time=self.max_task_time,
                timestamp=start_time,
                details={
                    "skill_count": len(skills),
                    "chain_execution_time": chain_time,
                    "average_skill_time": np.mean(skill_times) if skill_times else 0,
                    "max_skill_time": np.max(skill_times) if skill_times else 0,
                    "execution_success": execution_success,
                    "execution_message": execution_message,
                    "max_allowed_time": self.max_task_time,
                    "meets_time_requirement": success
                }
            )

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Error simulating complex task completion for '{task_name}': {e}")
            return TaskCompletionResult(
                test_name=f"Complex task completion test for '{task_name}'",
                task_name=task_name,
                success=False,
                total_time=total_time,
                skill_execution_times=skill_times,
                max_allowed_time=self.max_task_time,
                timestamp=start_time,
                details={"error": str(e)}
            )

    def run_task_completion_benchmark(self) -> List[TaskCompletionResult]:
        """
        Run comprehensive task completion benchmark

        Returns:
            List of task completion results
        """
        results = []
        num_iterations = self.config.get("num_test_iterations", 3)
        warmup_iterations = self.config.get("warmup_iterations", 1)

        self.logger.info(f"Running task completion benchmark with {num_iterations} iterations")

        # Warmup runs
        self.logger.info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            for task in self.test_tasks:
                self.simulate_task_completion(task["name"], task["command"])
                time.sleep(0.5)  # Small delay between tasks

        # Actual benchmark runs
        self.logger.info("Running actual benchmark iterations...")
        for iteration in range(num_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}")
            for task in self.test_tasks:
                result = self.simulate_task_completion(task["name"], task["command"])
                results.append(result)
                time.sleep(0.5)  # Small delay between tasks

        return results

    def run_complex_task_benchmark(self) -> List[TaskCompletionResult]:
        """
        Run benchmark with complex tasks involving multiple skills

        Returns:
            List of task completion results
        """
        results = []
        num_iterations = self.config.get("num_test_iterations", 3)
        warmup_iterations = self.config.get("warmup_iterations", 1)

        self.logger.info(f"Running complex task benchmark with {num_iterations} iterations")

        # Define complex tasks with multiple skills
        complex_tasks = [
            {
                "name": "tidy_apartment_full",
                "skills": self._create_tidy_apartment_skills()
            },
            {
                "name": "multi_object_transport",
                "skills": self._create_multi_object_transport_skills()
            }
        ]

        # Warmup runs
        self.logger.info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            for task in complex_tasks:
                self.simulate_complex_task(task["name"], task["skills"])
                time.sleep(0.5)

        # Actual benchmark runs
        self.logger.info("Running actual complex task benchmark iterations...")
        for iteration in range(num_iterations):
            self.logger.info(f"  Iteration {iteration + 1}/{num_iterations}")
            for task in complex_tasks:
                result = self.simulate_complex_task(task["name"], task["skills"])
                results.append(result)
                time.sleep(0.5)

        return results

    def _create_tidy_apartment_skills(self) -> List[RobotSkill]:
        """
        Create skills for a full apartment tidying task

        Returns:
            List of RobotSkill objects
        """
        from src.ros_interfaces.message_converters import RobotSkill

        skills = []

        # Navigate to living room
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "living room",
                "target_position": [0.0, 2.0, 0.0],
                "path_type": "shortest",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Grasp remote control
        skills.append(RobotSkill(
            skill_type="grasp",
            parameters={
                "target_object": "remote control",
                "grasp_type": "pinch",
                "grasp_position": [0.5, 1.5, 0.1],
                "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
            },
            confidence=0.8,
            execution_time=time.time(),
            status="pending"
        ))

        # Navigate to entertainment center
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "entertainment center",
                "target_position": [0.0, 1.0, 0.0],
                "path_type": "safe",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Place remote
        skills.append(RobotSkill(
            skill_type="place",
            parameters={
                "target_location": "entertainment center",
                "placement_type": "surface",
                "placement_position": [0.0, 1.0, 0.1]
            },
            confidence=0.82,
            execution_time=time.time(),
            status="pending"
        ))

        # Navigate to kitchen
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "kitchen",
                "target_position": [2.0, 0.0, 0.0],
                "path_type": "shortest",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Grasp coffee mug
        skills.append(RobotSkill(
            skill_type="grasp",
            parameters={
                "target_object": "coffee mug",
                "grasp_type": "power",
                "grasp_position": [2.0, 0.2, 0.1],
                "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
            },
            confidence=0.78,
            execution_time=time.time(),
            status="pending"
        ))

        # Navigate to sink
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "kitchen sink",
                "target_position": [2.5, 0.5, 0.0],
                "path_type": "safe",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Place mug in sink
        skills.append(RobotSkill(
            skill_type="place",
            parameters={
                "target_location": "kitchen sink",
                "placement_type": "surface",
                "placement_position": [2.5, 0.5, 0.1]
            },
            confidence=0.80,
            execution_time=time.time(),
            status="pending"
        ))

        return skills

    def _create_multi_object_transport_skills(self) -> List[RobotSkill]:
        """
        Create skills for a multi-object transport task

        Returns:
            List of RobotSkill objects
        """
        from src.ros_interfaces.message_converters import RobotSkill

        skills = []

        # Navigate to source location
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "dining table",
                "target_position": [1.0, 1.0, 0.0],
                "path_type": "shortest",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Grasp first object
        skills.append(RobotSkill(
            skill_type="grasp",
            parameters={
                "target_object": "plate",
                "grasp_type": "power",
                "grasp_position": [1.0, 1.2, 0.1],
                "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
            },
            confidence=0.8,
            execution_time=time.time(),
            status="pending"
        ))

        # Navigate to destination
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "kitchen counter",
                "target_position": [2.0, 0.0, 0.0],
                "path_type": "safe",
                "speed_profile": "slow"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Place first object
        skills.append(RobotSkill(
            skill_type="place",
            parameters={
                "target_location": "kitchen counter",
                "placement_type": "surface",
                "placement_position": [2.0, 0.0, 0.1]
            },
            confidence=0.82,
            execution_time=time.time(),
            status="pending"
        ))

        # Return to source for second object
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "dining table",
                "target_position": [1.0, 1.0, 0.0],
                "path_type": "shortest",
                "speed_profile": "normal"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Grasp second object
        skills.append(RobotSkill(
            skill_type="grasp",
            parameters={
                "target_object": "glass",
                "grasp_type": "pinch",
                "grasp_position": [1.2, 1.0, 0.1],
                "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
            },
            confidence=0.75,
            execution_time=time.time(),
            status="pending"
        ))

        # Navigate to destination again
        skills.append(RobotSkill(
            skill_type="navigate",
            parameters={
                "target_location": "kitchen counter",
                "target_position": [2.0, 0.2, 0.0],
                "path_type": "safe",
                "speed_profile": "slow"
            },
            confidence=0.85,
            execution_time=time.time(),
            status="pending"
        ))

        # Place second object
        skills.append(RobotSkill(
            skill_type="place",
            parameters={
                "target_location": "kitchen counter",
                "placement_type": "surface",
                "placement_position": [2.0, 0.2, 0.1]
            },
            confidence=0.80,
            execution_time=time.time(),
            status="pending"
        ))

        return skills

    def analyze_results(self, results: List[TaskCompletionResult]) -> Dict[str, Any]:
        """
        Analyze task completion test results

        Args:
            results: List of task completion results

        Returns:
            Analysis of the results
        """
        if not results:
            return {"error": "No results to analyze"}

        completion_times = [r.total_time for r in results]
        successful_completions = [r for r in results if r.success]
        failed_completions = [r for r in results if not r.success]

        # Calculate statistics
        avg_time = np.mean(completion_times) if completion_times else 0
        min_time = np.min(completion_times) if completion_times else 0
        max_time = np.max(completion_times) if completion_times else 0
        std_time = np.std(completion_times) if len(completion_times) > 1 else 0

        success_rate = len(successful_completions) / len(results) if results else 0

        # Analyze by task
        task_analysis = {}
        for task in self.test_tasks:
            task_results = [r for r in results if task["name"] in r.task_name]
            if task_results:
                task_times = [r.total_time for r in task_results]
                task_analysis[task["name"]] = {
                    "avg_time": np.mean(task_times),
                    "min_time": np.min(task_times),
                    "max_time": np.max(task_times),
                    "success_rate": len([r for r in task_results if r.success]) / len(task_results)
                }

        analysis = {
            "total_tests": len(results),
            "successful_completions": len(successful_completions),
            "failed_completions": len(failed_completions),
            "success_rate": success_rate,
            "time_stats": {
                "average": float(avg_time),
                "minimum": float(min_time),
                "maximum": float(max_time),
                "std_deviation": float(std_time),
                "max_allowed": self.max_task_time
            },
            "by_task": task_analysis,
            "meets_requirements": success_rate >= 0.8,  # Require 80% success rate
            "summary": f"Success rate: {success_rate:.1%} ({len(successful_completions)}/{len(results)} tasks completed under {self.max_task_time}s)"
        }

        return analysis

    def generate_completion_report(self, results: List[TaskCompletionResult]) -> str:
        """
        Generate a task completion report from the test results

        Args:
            results: List of task completion results

        Returns:
            Formatted completion report
        """
        analysis = self.analyze_results(results)

        report = []
        report.append("=" * 70)
        report.append("VLA System Task Completion Time Verification Report")
        report.append("=" * 70)
        report.append(f"Max Allowed Time: {self.max_task_time}s (3 minutes)")
        report.append(f"Target Time: {self.target_task_time}s (2 minutes)")
        report.append(f"Total Tests Run: {analysis['total_tests']}")
        report.append(f"Success Rate: {analysis['success_rate']:.1%}")
        report.append("")

        # Time statistics
        stats = analysis['time_stats']
        report.append("Completion Time Statistics:")
        report.append(f"  Average: {stats['average']:.3f}s")
        report.append(f"  Minimum: {stats['minimum']:.3f}s")
        report.append(f"  Maximum: {stats['maximum']:.3f}s")
        report.append(f"  Std Dev: {stats['std_deviation']:.3f}s")
        report.append("")

        # By task analysis
        report.append("Performance by Task:")
        for task_name, task_stats in analysis['by_task'].items():
            report.append(f"  {task_name}:")
            report.append(f"    Avg: {task_stats['avg_time']:.3f}s")
            report.append(f"    Success Rate: {task_stats['success_rate']:.1%}")
        report.append("")

        # Requirements check
        meets_req = analysis['meets_requirements']
        report.append(f"Requirements Check: {'PASS' if meets_req else 'FAIL'}")
        report.append(f"  > 80% tasks complete under {self.max_task_time}s: {'PASS' if meets_req else 'FAIL'}")
        report.append(f"  > Max time requirement ({self.max_task_time}s): {'PASS' if stats['maximum'] <= self.max_task_time else 'FAIL'}")
        report.append("")

        # Success criteria SC-001
        sc_001_met = analysis['success_rate'] >= 0.8 and stats['maximum'] <= self.max_task_time
        report.append(f"Success Criteria SC-001 Verification: {'PASS' if sc_001_met else 'FAIL'}")
        report.append(f"  SC-001: 80% success rate with <3 min completion time")
        report.append("=" * 70)

        return "\n".join(report)

    def run_complete_task_verification(self) -> Dict[str, Any]:
        """
        Run the complete task completion verification and return comprehensive results

        Returns:
            Dictionary with complete verification results
        """
        self.logger.info("Starting complete task completion verification...")

        # Run basic task benchmark
        basic_results = self.run_task_completion_benchmark()

        # Run complex task benchmark
        complex_results = self.run_complex_task_benchmark()

        # Combine all results
        all_results = basic_results + complex_results

        # Analyze results
        analysis = self.analyze_results(all_results)

        # Generate report
        report = self.generate_completion_report(all_results)

        complete_results = {
            "basic_tasks": {
                "results": basic_results,
                "analysis": self.analyze_results(basic_results)
            },
            "complex_tasks": {
                "results": complex_results,
                "analysis": self.analyze_results(complex_results)
            },
            "all_tasks": {
                "results": all_results,
                "analysis": analysis,
                "report": report
            },
            "sc_001_verified": (
                analysis['success_rate'] >= 0.8 and
                analysis['time_stats']['maximum'] <= self.max_task_time
            ),
            "summary": {
                "overall_success_rate": analysis['success_rate'],
                "max_completion_time": analysis['time_stats']['maximum'],
                "meets_sc_001": (
                    analysis['success_rate'] >= 0.8 and
                    analysis['time_stats']['maximum'] <= self.max_task_time
                )
            }
        }

        # Log summary
        self.logger.info(f"Overall success rate: {analysis['success_rate']:.1%}")
        self.logger.info(f"Max completion time: {analysis['time_stats']['maximum']:.3f}s")
        self.logger.info(f"SC-001 verified: {complete_results['sc_001_verified']}")

        return complete_results


def run_task_completion_tests():
    """
    Run the task completion tests and return results
    """
    logger = get_system_logger("TaskCompletionTestRunner")
    logger.info("Starting VLA System Task Completion Time Verification...")

    # Create the test suite
    test_suite = TaskCompletionTestSuite()

    # Run the complete verification
    results = test_suite.run_complete_task_verification()

    # Print report
    print("\n" + results["all_tasks"]["report"])

    # Print summary
    summary = results["summary"]
    print(f"\nVerification Summary:")
    print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
    print(f"  Max completion time: {summary['max_completion_time']:.3f}s")
    print(f"  Meets SC-001 criteria: {summary['meets_sc_001']}")

    return results


if __name__ == "__main__":
    results = run_task_completion_tests()
    print(f"\nTask completion verification completed.")
    print(f"SC-001 verification: {'PASS' if results['sc_001_verified'] else 'FAIL'}")