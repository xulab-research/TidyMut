from typing import (
    Any,
    Callable,
    List,
    Dict,
    Optional,
    Union,
    Tuple,
    NamedTuple,
    cast,
)
from copy import deepcopy
import logging
from functools import wraps
import time
import pickle
from dataclasses import dataclass


class MultiOutput(NamedTuple):
    """Container for functions that return multiple outputs"""

    main: Any  # Main data to pass to next step
    side: Dict[str, Any] = {}  # Side outputs to store


@dataclass
class PipelineOutput:
    """Structured output from pipeline steps"""

    data: Any  # Main data flow
    artifacts: Dict[str, Any]  # Named artifacts/side outputs

    def __getitem__(self, key: str):
        """Allow dictionary-style access to artifacts"""
        return self.artifacts.get(key)


class PipelineStep:
    """Represents a single step in the pipeline"""

    def __init__(self, name: str, function: Callable, *args, **kwargs):
        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.side_outputs = {}  # Store side outputs
        self.execution_time = None
        self.success = False
        self.error = None

    def execute(self, data: Any) -> Any:
        """Execute this pipeline step"""
        start_time = time.time()

        try:
            # Apply function with args and kwargs - always pass actual data
            result = self.function(data, *self.args, **self.kwargs)

            # Handle different output types based on step type
            step_type = getattr(self.function, "_step_type", "unknown")

            if isinstance(result, MultiOutput):
                # Result from @multiout_step decorated function
                self.result = result.main
                self.side_outputs = result.side
                final_result = result.main

            elif step_type == "multi_output":
                # This shouldn't happen if @multiout_step is working correctly
                raise RuntimeError(
                    f"Function {self.function.__name__} is marked as multi_output "
                    f"but didn't return MultiOutput. This indicates a decorator bug."
                )

            elif step_type == "single_output":
                # Function decorated with @pipeline_step - treat any result as single value
                self.result = result
                final_result = result

            else:
                # Undecorated function - treat as single output with warning
                import warnings

                warnings.warn(
                    f"Function {getattr(self.function, '__name__', 'unknown')} is not "
                    f"decorated with @pipeline_step or @multiout_step. "
                    f"Consider adding @pipeline_step for better pipeline integration.",
                    UserWarning,
                )
                self.result = result
                final_result = result

            self.success = True
            return final_result

        except Exception as e:
            self.success = False
            self.error = e
            raise
        finally:
            self.execution_time = time.time() - start_time

    def get_step_info(self) -> Dict[str, Any]:
        """Get detailed information about this step"""
        return {
            "name": self.name,
            "step_type": getattr(self.function, "_step_type", "unknown"),
            "expected_outputs": getattr(self.function, "_expected_output_count", 1),
            "output_names": getattr(self.function, "_output_names", ["main"]),
            "success": self.success,
            "execution_time": self.execution_time,
            "has_side_outputs": bool(self.side_outputs),
            "side_output_keys": (
                list(self.side_outputs.keys()) if self.side_outputs else []
            ),
            "error": str(self.error) if self.error else None,
        }


class Pipeline:
    """Pipeline for processing data with pandas-style method chaining"""

    def __init__(
        self, data: Any = None, name: Optional[str] = None, logging_level: str = "INFO"
    ):
        self.name = name or "Pipeline"
        self._data = data  # Store actual data
        self._artifacts: Dict[str, Any] = {}  # Store artifacts separately
        self.steps: List[PipelineStep] = []
        self.results: List[Any] = []

        # Setup logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, logging_level))

        # Add handler if logger doesn't have one
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @property
    def data(self) -> Any:
        """
        Always return the actual data, never PipelineOutput.

        This ensures consistent user experience - pipeline.data can always be
        used with methods like .copy(), .append(), etc.
        """
        if isinstance(self._data, PipelineOutput):
            return self._data.data
        return self._data

    @property
    def artifacts(self) -> Dict[str, Any]:
        """
        Always return the artifacts dictionary.

        This provides direct access to all stored artifacts from pipeline steps.
        """
        if isinstance(self._data, PipelineOutput):
            return self._data.artifacts
        return self._artifacts

    @property
    def structured_data(self) -> PipelineOutput:
        """
        Return PipelineOutput object with both data and artifacts.

        Use this when you need the complete pipeline state for serialization,
        passing to other systems, or when working with structured data flows.
        """
        if isinstance(self._data, PipelineOutput):
            return self._data
        return PipelineOutput(data=self._data, artifacts=self._artifacts.copy())

    def then(self, func: Callable, *args, **kwargs) -> "Pipeline":
        """Apply a function to the current data (pandas.pipe style)"""
        # Use custom step name if available from decorator
        if hasattr(func, "_step_name"):
            step_name = func._step_name
        else:
            step_name = getattr(func, "__name__", str(func))

        if self._data is None:
            raise ValueError("No data to process. Initialize pipeline with data.")

        # Validate if function is marked as pipeline step
        if hasattr(func, "_is_pipeline_step") and func._is_pipeline_step:
            self.logger.debug(f"Executing pipeline step: {step_name}")

        # Create and execute step
        step = PipelineStep(step_name, func, *args, **kwargs)
        self.steps.append(step)

        self.logger.info(f"Executing step: {step_name}")

        try:
            # Execute step - always pass actual data to function
            result = step.execute(self.data)

            # Update internal data
            self._data = result

            # Store side outputs in artifacts
            if step.side_outputs:
                for key, value in step.side_outputs.items():
                    artifact_key = f"{step_name}.{key}" if key else step_name
                    self._artifacts[artifact_key] = value

            # Store result for history
            self.results.append(self.data)

            self.logger.info(
                f"Step '{step_name}' completed in {step.execution_time:.3f}s"
            )

            # Log side outputs if any
            if step.side_outputs:
                self.logger.info(
                    f"Step '{step_name}' produced {len(step.side_outputs)} side outputs"
                )

        except Exception as e:
            self.logger.error(f"Step '{step_name}' failed: {str(e)}")
            raise RuntimeError(
                f"Pipeline failed at step '{step_name}': {str(e)}"
            ) from e

        return self

    def apply(self, func: Callable, *args, **kwargs) -> "Pipeline":
        """Apply function and return new Pipeline (functional style)"""
        new_pipeline = self.copy()
        return new_pipeline.then(func, *args, **kwargs)

    def assign(self, **kwargs) -> "Pipeline":
        """Add attributes or computed values to data"""

        def assign_values(data):
            # Handle different data types
            if hasattr(data, "__dict__"):
                for key, value in kwargs.items():
                    setattr(data, key, value)
            elif isinstance(data, dict):
                data.update(kwargs)
            else:
                # For immutable types, wrap in a container
                return {"data": data, **kwargs}
            return data

        return self.then(assign_values)

    def filter(self, condition: Callable) -> "Pipeline":
        """Filter data based on condition"""

        def filter_data(data):
            if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                return type(data)(item for item in data if condition(item))
            elif condition(data):
                return data
            else:
                raise ValueError("Data does not meet filter condition")

        return self.then(filter_data)

    def transform(self, transformer: Callable, *args, **kwargs) -> "Pipeline":
        """Alias of `then`, used to define format transformations."""
        return self.then(transformer, *args, **kwargs)

    def validate(
        self, validator: Callable, error_msg: str = "Validation failed"
    ) -> "Pipeline":
        """Validate data and raise error if invalid"""

        def validate_data(data):
            if not validator(data):
                raise ValueError(error_msg)
            return data

        return self.then(validate_data)

    def peek(self, func: Optional[Callable] = None, prefix: str = "") -> "Pipeline":
        """Inspect data without modifying it (for debugging)"""

        def peek_data(data):
            if func:
                func(data)
            else:
                msg = (
                    f"{prefix}Pipeline data: {repr(data)}"
                    if prefix
                    else f"Pipeline data: {repr(data)}"
                )
                self.logger.debug(msg)
            return data

        return self.then(peek_data)

    def store(self, name: str, extractor: Optional[Callable] = None) -> "Pipeline":
        """Store current data or extracted value as artifact"""

        def store_data(data):
            if extractor:
                self._artifacts[name] = extractor(data)
            else:
                self._artifacts[name] = deepcopy(data)
            return data

        return self.then(store_data)

    def copy(self) -> "Pipeline":
        """Create a deep copy of this pipeline"""
        new_pipeline = Pipeline(
            deepcopy(self.data),  # Always copy actual data
            f"{self.name}_copy",
            logging_level=logging.getLevelName(self.logger.level),
        )
        # Copy artifacts but not steps and results
        new_pipeline._artifacts = deepcopy(self.artifacts)
        return new_pipeline

    def get_data(self) -> Any:
        """
        Get current data (same as .data property).

        Kept for backward compatibility.
        """
        return self.data

    def get_artifact(self, name: str) -> Any:
        """Get a specific artifact by name"""
        if name in self.artifacts:
            return self.artifacts[name]
        else:
            raise KeyError(
                f"Artifact '{name}' not found. Available: {list(self.artifacts.keys())}"
            )

    def get_all_artifacts(self) -> Dict[str, Any]:
        """Get all stored artifacts"""
        return self.artifacts.copy()

    def get_step_result(self, step_index: Union[int, str]) -> Any:
        """Get result from a specific step by index or name"""
        if isinstance(step_index, int):
            if 0 <= step_index < len(self.results):
                return self.results[step_index]
            else:
                raise IndexError(f"Step index {step_index} out of range")
        else:
            for i, step in enumerate(self.steps):
                if step.name == step_index:
                    if i < len(self.results):
                        return self.results[i]
                    else:
                        raise ValueError(
                            f"Step '{step_index}' has not completed execution"
                        )
            raise ValueError(f"Step '{step_index}' not found")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        summary = {
            "pipeline_name": self.name,
            "total_steps": len(self.steps),
            "successful_steps": sum(1 for s in self.steps if s.success),
            "failed_steps": sum(1 for s in self.steps if s.error is not None),
            "total_execution_time": sum(s.execution_time or 0 for s in self.steps),
            "artifacts_count": len(self.artifacts),
            "steps": [],
        }

        for i, step in enumerate(self.steps):
            step_info = {
                "index": i,
                "name": step.name,
                "success": step.success,
                "execution_time": step.execution_time,
                "error": str(step.error) if step.error else None,
                "side_outputs": (
                    list(step.side_outputs.keys()) if step.side_outputs else []
                ),
            }
            summary["steps"].append(step_info)

        return summary

    def visualize_pipeline(self) -> str:
        """Generate a text visualization of the pipeline"""
        lines = [f"Pipeline: {self.name}", "=" * 40]

        for i, step in enumerate(self.steps):
            status = "✓" if step.success else "✗" if step.error else "○"
            time_str = (
                f"({step.execution_time:.3f}s)" if step.execution_time else "(pending)"
            )

            # Check if it's a decorated pipeline step
            if hasattr(step.function, "_is_pipeline_step"):
                lines.append(f"{status} Step {i+1}: {step.name} {time_str} [validated]")
            else:
                lines.append(f"{status} Step {i+1}: {step.name} {time_str}")

            # Add description if available
            if (
                hasattr(step.function, "_step_description")
                and step.function._step_description
            ):
                lines.append(f"   └─ {step.function._step_description.strip()}")

            # Show side outputs
            if step.side_outputs:
                for key in step.side_outputs:
                    lines.append(f"   └─ side output: {key}")

        lines.append("=" * 40)
        lines.append(f"Current data type: {type(self.data).__name__}")
        lines.append(f"Artifacts stored: {len(self.artifacts)}")

        return "\n".join(lines)

    def save(self, filepath: str, format: str = "pickle") -> "Pipeline":
        """Save current data to file"""
        data_to_save = self.data

        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data_to_save, f)
        elif format == "json":
            import json

            with open(filepath, "w") as f:
                json.dump(data_to_save, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Pipeline data saved to {filepath}")
        return self

    def save_artifacts(self, filepath: str, format: str = "pickle") -> "Pipeline":
        """Save all artifacts to file"""
        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self.artifacts, f)
        elif format == "json":
            import json

            with open(filepath, "w") as f:
                json.dump(self.artifacts, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Pipeline artifacts saved to {filepath}")
        return self

    def save_structured_data(self, filepath: str, format: str = "pickle") -> "Pipeline":
        """Save structured data (data + artifacts) to file"""
        structured = self.structured_data

        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(structured, f)
        elif format == "json":
            import json

            # Convert to dict for JSON serialization
            data_dict = {"data": structured.data, "artifacts": structured.artifacts}
            with open(filepath, "w") as f:
                json.dump(data_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Pipeline structured data saved to {filepath}")
        return self

    @classmethod
    def load(
        cls, filepath: str, format: str = "pickle", name: Optional[str] = None
    ) -> "Pipeline":
        """Load data from file and create new pipeline"""
        if format == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        elif format == "json":
            import json

            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return cls(data, name or f"Pipeline_from_{filepath}")

    @classmethod
    def load_structured_data(
        cls, filepath: str, format: str = "pickle", name: Optional[str] = None
    ) -> "Pipeline":
        """Load structured data from file and create new pipeline"""
        if format == "pickle":
            with open(filepath, "rb") as f:
                structured = pickle.load(f)

            if isinstance(structured, PipelineOutput):
                pipeline = cls(structured.data, name)
                pipeline._artifacts = structured.artifacts
                return pipeline
            else:
                return cls(structured, name)

        elif format == "json":
            import json

            with open(filepath, "r") as f:
                data_dict = json.load(f)

            if (
                isinstance(data_dict, dict)
                and "data" in data_dict
                and "artifacts" in data_dict
            ):
                pipeline = cls(data_dict["data"], name)
                pipeline._artifacts = data_dict["artifacts"]
                return pipeline
            else:
                return cls(data_dict, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def __str__(self) -> str:
        success_count = sum(1 for s in self.steps if s.success)
        artifacts_str = f", {len(self.artifacts)} artifacts" if self.artifacts else ""
        return f"Pipeline('{self.name}'): {success_count}/{len(self.steps)} steps executed{artifacts_str}"

    def __repr__(self) -> str:
        return f"<Pipeline name='{self.name}' steps={len(self.steps)} data_type={type(self.data).__name__} artifacts={len(self.artifacts)}>"


def create_pipeline(data: Any, name: Optional[str] = None, **kwargs) -> Pipeline:
    """Create a new pipeline with initial data"""
    return Pipeline(data, name, **kwargs)


def pipeline_step(name: Union[str, Callable[..., Any], None] = None):
    """
    Decorator for single-output pipeline functions.

    Use this for functions that return a single value (including tuples as single values).
    For multiple outputs, use @multiout_step instead.

    Parameters
    ----------
    name : Optional[str] or Callable
        Custom name for the step. If None, uses function name.
        When used as @pipeline_step (without parentheses), this will be the function.

    Examples
    --------
    >>> @pipeline_step
    ... def process(data):
    ...     return processed_data  # Single output

    >>> @pipeline_step("process_data")
    ... def process(data):
    ...     return processed_data  # Single output

    >>> @pipeline_step()
    ... def get_coordinates():
    ...     return (10, 20)  # Single tuple output
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add consistent metadata attributes
        setattr(wrapper, "_is_pipeline_step", True)
        setattr(wrapper, "_step_type", "single_output")
        setattr(wrapper, "_step_name", name if isinstance(name, str) else func.__name__)
        setattr(wrapper, "_step_description", func.__doc__)
        setattr(wrapper, "_expected_output_count", 1)
        setattr(wrapper, "_output_names", ["main"])

        return wrapper

    # Handle both @pipeline_step and @pipeline_step() usage
    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


def multiout_step(**outputs: str):
    """
    Decorator for multi-output pipeline functions.

    Use this for functions that return multiple values where you want
    to name and access the outputs separately.

    Parameters
    ----------
    **outputs : str
        Named outputs. Use 'main' to specify which output is the main data flow.
        If 'main' is not specified, the first return value is treated as main.

    Examples
    --------
    >>> # Returns 3 values: main, stats, plot
    >>> @multiout_step(stats="statistics", plot="visualization")
    ... def analyze_data(data):
    ...     ...
    ...     return processed_data, stats_dict, plot_object

    >>> # Returns 3 values with explicit main designation
    >>> @multiout_step(main="result", error="error_info", stats="statistics")
    ... def process_with_metadata(data):
    ...     ...
    ...     return result, error_info, stats

    Note
    ----
    With this decorator, side outputs are returned as a dictionary.
    """
    has_explicit_main = "main" in outputs
    side_output_items = [(k, v) for k, v in outputs.items() if k != "main"]
    side_output_names = [v for _, v in side_output_items]

    # Calculate expected number of return values
    expected_count = len(outputs) if has_explicit_main else len(side_output_items) + 1

    def decorator(func: Callable[..., Tuple[Any, ...]]) -> Callable[..., MultiOutput]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> MultiOutput:
            results = func(*args, **kwargs)

            # Validate return type and count
            if not isinstance(results, tuple):
                if expected_count > 1:
                    raise ValueError(
                        f"Function {func.__name__} decorated with @multiout_step "
                        f"expected {expected_count} return values but got 1 non-tuple value. "
                        f"For single outputs, use @pipeline_step instead."
                    )
                return MultiOutput(main=results, side={})

            # Check if tuple length matches expected outputs
            if len(results) != expected_count:
                raise ValueError(
                    f"Function {func.__name__} decorated with @multiout_step "
                    f"expected {expected_count} return values but got {len(results)}. "
                    f"Declared outputs: {list(outputs.keys())}"
                )

            # Process as multiple values
            if has_explicit_main:
                main_index = list(outputs.keys()).index("main")
                main = results[main_index]
                side = {}
                for i, (_, name) in enumerate(side_output_items):
                    actual_index = i if i < main_index else i + 1
                    if actual_index < len(results):
                        side[name] = results[actual_index]
            else:
                main = results[0]
                side = {}
                for i, (_, name) in enumerate(side_output_items):
                    if i + 1 < len(results):
                        side[name] = results[i + 1]

            return MultiOutput(main=main, side=side)

        # Add consistent metadata attributes
        setattr(wrapper, "_is_pipeline_step", True)
        setattr(wrapper, "_step_type", "multi_output")
        setattr(wrapper, "_step_name", func.__name__)
        setattr(wrapper, "_step_description", func.__doc__)
        setattr(wrapper, "_expected_output_count", expected_count)
        setattr(wrapper, "_output_names", ["main"] + side_output_names)
        setattr(wrapper, "_declared_outputs", outputs)

        return cast(Callable[..., MultiOutput], wrapper)

    return decorator


# Helper function to create multi-output values directly
def create_multi_output(main_data, **side_outputs: Any) -> MultiOutput:
    """
    Helper function to create MultiOutput values directly.

    Args:
        main_data: The main data to pass to next pipeline step
        **side_outputs: Named side outputs to store as artifacts

    Returns:
        MultiOutput object with main and side data
    """
    return MultiOutput(main=main_data, side=side_outputs)
