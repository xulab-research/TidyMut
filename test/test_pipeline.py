import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
import json
import pickle

from tidymut.core.pipeline import (
    Pipeline,
    PipelineStep,
    create_pipeline,
    pipeline_step,
    MultiOutput,
    PipelineOutput,
    multiout_step,
)


class TestSimplifiedPipelineInit:
    """测试简化版Pipeline初始化"""

    def test_pipeline_creation_with_data(self):
        """测试使用数据创建pipeline"""
        data = [1, 2, 3]
        pipeline = Pipeline(data, name="test_pipeline")

        assert pipeline.name == "test_pipeline"
        assert pipeline.data == [1, 2, 3]  # 总是返回实际数据
        assert pipeline.artifacts == {}  # 总是返回字典
        assert pipeline.steps == []
        assert pipeline.results == []

    def test_pipeline_creation_without_data(self):
        """测试不使用数据创建pipeline"""
        pipeline = Pipeline(name="empty_pipeline")

        assert pipeline.name == "empty_pipeline"
        assert pipeline.data is None
        assert pipeline.artifacts == {}

    def test_create_pipeline_helper(self):
        """测试create_pipeline辅助函数"""
        data = {"key": "value"}
        pipeline = create_pipeline(data, name="helper_test")

        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "helper_test"
        assert pipeline.data == {"key": "value"}


class TestSimplifiedPipelineProperties:
    """测试简化版Pipeline的属性"""

    def test_data_property_always_returns_actual_data(self):
        """测试data属性总是返回实际数据"""
        # 初始化时
        pipeline = Pipeline([1, 2, 3])
        assert pipeline.data == [1, 2, 3]
        assert isinstance(pipeline.data, list)

        # 可以直接调用list方法
        copied_data = pipeline.data.copy()
        assert copied_data == [1, 2, 3]

        # 数据变化后
        def double(data):
            return [x * 2 for x in data]

        pipeline.then(double)
        assert pipeline.data == [2, 4, 6]
        assert isinstance(pipeline.data, list)

    def test_artifacts_property_always_returns_dict(self):
        """测试artifacts属性总是返回字典"""
        pipeline = Pipeline([1, 2, 3])

        # 初始时为空字典
        assert pipeline.artifacts == {}
        assert isinstance(pipeline.artifacts, dict)

        # 添加artifact后
        @multiout_step(stats="statistics")
        def analyze(data):
            return [x * 2 for x in data], {"count": len(data)}

        pipeline.then(analyze)

        assert "analyze.statistics" in pipeline.artifacts
        assert pipeline.artifacts["analyze.statistics"]["count"] == 3
        assert isinstance(pipeline.artifacts, dict)

    def test_structured_data_property_returns_pipeline_output(self):
        """测试structured_data属性返回PipelineOutput对象"""
        pipeline = Pipeline([1, 2, 3])

        # 初始时
        structured = pipeline.structured_data
        assert isinstance(structured, PipelineOutput)
        assert structured.data == [1, 2, 3]
        assert structured.artifacts == {}

        # 添加artifact后
        @multiout_step(stats="statistics")
        def analyze(data):
            return [x * 2 for x in data], {"count": len(data)}

        pipeline.then(analyze)

        structured = pipeline.structured_data
        assert isinstance(structured, PipelineOutput)
        assert structured.data == [2, 4, 6]
        assert "analyze.statistics" in structured.artifacts

    def test_consistent_user_experience(self):
        """测试一致的用户体验"""
        pipeline = Pipeline([1, 2, 3])

        # 这些操作在任何时候都应该工作
        original_data = pipeline.data.copy()  # ✓ 总是可以调用
        artifacts_keys = list(pipeline.artifacts.keys())  # ✓ 总是可以调用

        @multiout_step(stats="statistics")
        def analyze(data):
            return [x * 2 for x in data], {"count": len(data)}

        pipeline.then(analyze)

        # 执行后依然可以使用相同的方式访问
        new_data = pipeline.data.copy()  # ✓ 依然可以调用
        artifacts_keys = list(pipeline.artifacts.keys())  # ✓ 依然可以调用

        assert len(new_data) == 3
        assert len(artifacts_keys) == 1


class TestSimplifiedPipelineBasicOperations:
    """测试简化版Pipeline基本操作"""

    def test_then_with_simple_function(self):
        """测试then方法与简单函数"""

        def double(x):
            return [item * 2 for item in x]

        pipeline = Pipeline([1, 2, 3])
        result_pipeline = pipeline.then(double)

        assert result_pipeline is pipeline  # 返回自身
        assert pipeline.data == [2, 4, 6]
        assert len(pipeline.steps) == 1
        assert len(pipeline.results) == 1
        assert pipeline.steps[0].success is True

    def test_then_chaining(self):
        """测试链式调用"""

        def add_one(data):
            return [x + 1 for x in data]

        def multiply_by_two(data):
            return [x * 2 for x in data]

        pipeline = Pipeline([1, 2, 3])
        result = pipeline.then(add_one).then(multiply_by_two)

        assert result is pipeline
        assert pipeline.data == [4, 6, 8]  # (1+1)*2, (2+1)*2, (3+1)*2
        assert len(pipeline.steps) == 2
        assert all(step.success for step in pipeline.steps)

    def test_pipeline_step_decorator_integration(self):
        """测试@pipeline_step装饰器集成"""

        @pipeline_step("process_data")
        def process_list(data):
            """Process the input list"""
            return [x * 3 for x in data]

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(process_list)

        assert pipeline.data == [3, 6, 9]
        assert pipeline.steps[0].name == "process_data"
        assert hasattr(process_list, "_is_pipeline_step")
        assert getattr(process_list, "_step_type") == "single_output"

    def test_multiout_step_decorator_integration(self):
        """测试@multiout_step装饰器集成"""

        @multiout_step(stats="statistics", metadata="info")
        def analyze_data(data):
            """Analyze data with statistics"""
            processed = [x * 2 for x in data]
            stats = {"count": len(processed), "sum": sum(processed)}
            metadata = {"processed_at": "2024-01-01"}
            return processed, stats, metadata

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(analyze_data)

        # 主数据流
        assert pipeline.data == [2, 4, 6]

        # Side outputs存储在artifacts中
        assert "analyze_data.statistics" in pipeline.artifacts
        assert "analyze_data.info" in pipeline.artifacts
        assert pipeline.artifacts["analyze_data.statistics"]["count"] == 3
        assert pipeline.artifacts["analyze_data.info"]["processed_at"] == "2024-01-01"

    def test_multiout_step_with_explicit_main(self):
        """测试带显式main的@multiout_step"""

        @multiout_step(main="result", errors="error_list", warnings="warning_list")
        def comprehensive_process(data):
            result = [x + 1 for x in data]
            errors = []
            warnings = ["minor issue"]
            return result, errors, warnings

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(comprehensive_process)

        assert pipeline.data == [2, 3, 4]
        assert "comprehensive_process.error_list" in pipeline.artifacts
        assert "comprehensive_process.warning_list" in pipeline.artifacts
        assert pipeline.artifacts["comprehensive_process.error_list"] == []
        assert pipeline.artifacts["comprehensive_process.warning_list"] == [
            "minor issue"
        ]


class TestSimplifiedPipelineUtilityMethods:
    """测试简化版Pipeline工具方法"""

    def test_peek_method(self):
        """测试peek方法"""

        def double_data(data):
            return [x * 2 for x in data]

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(double_data)

        # 记录数据状态用于对比
        original_data = pipeline.data.copy()  # ✓ 现在这个总是可以工作
        result = pipeline.peek()

        assert result is pipeline
        assert pipeline.data == original_data  # 数据未被修改
        assert pipeline.data == [2, 4, 6]

    def test_peek_with_custom_function(self):
        """测试peek方法与自定义函数"""
        captured_data = []

        def capture_data(data):
            captured_data.append(data.copy())

        pipeline = Pipeline([1, 2, 3])
        pipeline.peek(capture_data)

        assert captured_data == [[1, 2, 3]]
        assert pipeline.data == [1, 2, 3]  # 数据未改变

    def test_store_method(self):
        """测试store方法存储artifact"""

        def double_data(data):
            return [x * 2 for x in data]

        def extract_sum(data):
            return sum(data)

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(double_data).store("doubled_sum", extract_sum)

        assert "doubled_sum" in pipeline.artifacts
        assert pipeline.artifacts["doubled_sum"] == 12  # sum([2, 4, 6])

    def test_get_artifact_method(self):
        """测试get_artifact方法"""

        @multiout_step(stats="statistics")
        def process_with_stats(data):
            return [x * 2 for x in data], {"count": len(data)}

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(process_with_stats)

        stats = pipeline.get_artifact("process_with_stats.statistics")
        assert stats == {"count": 3}

        with pytest.raises(KeyError, match="Artifact 'nonexistent' not found"):
            pipeline.get_artifact("nonexistent")

    def test_get_all_artifacts_method(self):
        """测试get_all_artifacts方法"""

        @multiout_step(stats="statistics", info="metadata")
        def process_data(data):
            return data, {"count": len(data)}, {"type": "test"}

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(process_data)

        all_artifacts = pipeline.get_all_artifacts()

        assert isinstance(all_artifacts, dict)
        assert "process_data.statistics" in all_artifacts
        assert "process_data.metadata" in all_artifacts
        assert all_artifacts["process_data.statistics"]["count"] == 3

        # 验证返回的是副本
        all_artifacts["new_key"] = "new_value"
        assert "new_key" not in pipeline.artifacts

    def test_copy_method(self):
        """测试copy方法"""

        @multiout_step(stats="statistics")
        def process_data(data):
            return [x * 2 for x in data], {"count": len(data)}

        original = Pipeline([1, 2, 3], name="original")
        original.then(process_data)

        copied = original.copy()

        assert copied is not original
        assert copied.name == "original_copy"
        assert copied.data == [2, 4, 6]  # 数据被复制
        assert copied.artifacts == original.artifacts  # artifacts被复制
        assert copied.steps == []  # Steps不被复制
        assert copied.results == []  # Results不被复制

    def test_get_execution_summary(self):
        """测试获取执行摘要"""

        @pipeline_step("step1")
        def successful_step(data):
            return [x + 1 for x in data]

        def failing_step(data):
            raise ValueError("Test error")

        pipeline = Pipeline([1, 2, 3], name="test_pipeline")
        pipeline.then(successful_step)

        try:
            pipeline.then(failing_step)
        except RuntimeError:
            pass

        summary = pipeline.get_execution_summary()

        assert summary["pipeline_name"] == "test_pipeline"
        assert summary["total_steps"] == 2
        assert summary["successful_steps"] == 1
        assert summary["failed_steps"] == 1
        assert summary["total_execution_time"] > 0
        assert len(summary["steps"]) == 2
        assert summary["steps"][0]["success"] is True
        assert summary["steps"][1]["success"] is False

    def test_visualize_pipeline(self):
        """测试pipeline可视化"""

        @pipeline_step("process_data")
        def process_data(data):
            """Process the input data"""
            return [x * 2 for x in data]

        pipeline = Pipeline([1, 2, 3], name="viz_test")
        pipeline.then(process_data)

        visualization = pipeline.visualize_pipeline()

        assert "Pipeline: viz_test" in visualization
        assert "✓ Step 1: process_data" in visualization
        assert "[validated]" in visualization
        assert "Process the input data" in visualization
        assert "Current data type: list" in visualization

    def test_str_and_repr_methods(self):
        """测试字符串表示方法"""

        @multiout_step(stats="statistics")
        def process_data(data):
            return data, {"count": len(data)}

        pipeline = Pipeline([1, 2, 3], name="test_pipe")
        pipeline.then(process_data)

        str_repr = str(pipeline)
        assert "Pipeline('test_pipe')" in str_repr
        assert "1/1 steps executed" in str_repr
        assert "1 artifacts" in str_repr

        repr_str = repr(pipeline)
        assert "<Pipeline name='test_pipe'" in repr_str
        assert "steps=1" in repr_str
        assert "data_type=list" in repr_str
        assert "artifacts=1" in repr_str


class TestSimplifiedPipelineFileOperations:
    """测试简化版Pipeline文件操作"""

    def test_save_and_load_data(self):
        """测试数据的保存和加载"""
        pipeline = Pipeline([1, 2, 3], name="save_test")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # 保存数据
            result = pipeline.save(temp_path, format="pickle")
            assert result is pipeline

            # 加载数据
            loaded_pipeline = Pipeline.load(
                temp_path, format="pickle", name="loaded_test"
            )
            assert loaded_pipeline.data == [1, 2, 3]
            assert loaded_pipeline.name == "loaded_test"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_and_load_artifacts(self):
        """测试artifacts的保存和加载"""

        @multiout_step(stats="statistics")
        def process_data(data):
            return data, {"count": len(data), "sum": sum(data)}

        pipeline = Pipeline([1, 2, 3])
        pipeline.then(process_data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            result = pipeline.save_artifacts(temp_path, format="pickle")
            assert result is pipeline

            # 验证artifacts被保存
            with open(temp_path, "rb") as f:
                loaded_artifacts = pickle.load(f)

            assert "process_data.statistics" in loaded_artifacts
            assert loaded_artifacts["process_data.statistics"]["count"] == 3

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_and_load_structured_data(self):
        """测试structured data的保存和加载"""

        @multiout_step(stats="statistics")
        def process_data(data):
            return [x * 2 for x in data], {"count": len(data)}

        pipeline = Pipeline([1, 2, 3], name="structured_test")
        pipeline.then(process_data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # 保存structured data
            result = pipeline.save_structured_data(temp_path, format="pickle")
            assert result is pipeline

            # 加载structured data
            loaded_pipeline = Pipeline.load_structured_data(
                temp_path, format="pickle", name="loaded_structured"
            )
            assert loaded_pipeline.data == [2, 4, 6]
            assert "process_data.statistics" in loaded_pipeline.artifacts
            assert loaded_pipeline.artifacts["process_data.statistics"]["count"] == 3

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_structured_data_json(self):
        """测试JSON格式的structured data保存和加载"""
        pipeline = Pipeline({"numbers": [1, 2, 3]}, name="json_test")
        pipeline.store("metadata", lambda data: {"type": "dict"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # 保存为JSON
            pipeline.save_structured_data(temp_path, format="json")

            # 加载JSON
            loaded_pipeline = Pipeline.load_structured_data(temp_path, format="json")
            assert loaded_pipeline.data == {"numbers": [1, 2, 3]}
            assert "metadata" in loaded_pipeline.artifacts
            assert loaded_pipeline.artifacts["metadata"]["type"] == "dict"

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSimplifiedPipelineAdvancedFeatures:
    """测试简化版Pipeline高级功能"""

    def test_filter_method(self):
        """测试filter方法"""
        pipeline = Pipeline([1, 2, 3, 4, 5])

        # 过滤出偶数
        result = pipeline.filter(lambda x: x % 2 == 0)

    def test_assign_method_with_dict(self):
        """测试assign方法与字典数据"""
        pipeline = Pipeline({"a": 1, "b": 2})
        result = pipeline.assign(c=3, d=4)

        assert result is pipeline
        assert pipeline.data == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_validate_method_success(self):
        """测试validate方法成功场景"""

        def is_positive_list(data):
            return all(x > 0 for x in data)

        pipeline = Pipeline([1, 2, 3])
        result = pipeline.validate(is_positive_list, "All numbers must be positive")

        assert result is pipeline
        assert pipeline.data == [1, 2, 3]

    def test_validate_method_failure(self):
        """测试validate方法失败场景"""

        def is_positive_list(data):
            return all(x > 0 for x in data)

        pipeline = Pipeline([-1, 2, 3])

        with pytest.raises(RuntimeError, match="All numbers must be positive"):
            pipeline.validate(is_positive_list, "All numbers must be positive")

    def test_apply_method_functional_style(self):
        """测试apply方法的函数式风格"""

        def double(data):
            return [x * 2 for x in data]

        original = Pipeline([1, 2, 3])
        new_pipeline = original.apply(double)

        # 原pipeline不变
        assert original.data == [1, 2, 3]

        # 新pipeline有变化
        assert new_pipeline.data == [2, 4, 6]
        assert new_pipeline is not original
        assert new_pipeline.name == "Pipeline_copy"

    def test_complex_pipeline_workflow(self):
        """测试复杂的pipeline工作流"""

        @pipeline_step("normalize")
        def normalize(data):
            """Normalize data to 0-1 range"""
            max_val = max(data)
            return [x / max_val for x in data]

        @multiout_step(stats="statistics", summary="summary")
        def analyze(data):
            """Analyze normalized data"""
            processed = [round(x, 2) for x in data]
            stats = {"mean": sum(data) / len(data), "max": max(data), "min": min(data)}
            summary = f"Processed {len(data)} items"
            return processed, stats, summary

        # 构建复杂pipeline
        pipeline = (
            Pipeline([10, 20, 30, 40, 50], name="complex_workflow")
            .then(normalize)
            .store("normalized", lambda data: data.copy())
            .then(analyze)
            .store("final_result")
        )

        # 验证主数据流
        assert pipeline.data == [0.2, 0.4, 0.6, 0.8, 1.0]

        # 验证artifacts
        assert "normalized" in pipeline.artifacts
        assert "analyze.statistics" in pipeline.artifacts
        assert "analyze.summary" in pipeline.artifacts
        assert "final_result" in pipeline.artifacts

        # 验证统计数据
        stats = pipeline.artifacts["analyze.statistics"]
        assert stats["max"] == 1.0
        assert stats["min"] == 0.2

        # 验证structured data
        structured = pipeline.structured_data
        assert isinstance(structured, PipelineOutput)
        assert structured.data == pipeline.data
        assert len(structured.artifacts) == 4

        # 验证可视化
        viz = pipeline.visualize_pipeline()
        assert "complex_workflow" in viz
        assert "normalize" in viz
        assert "analyze" in viz


class TestSimplifiedPipelineErrorHandling:
    """测试简化版Pipeline错误处理"""

    def test_step_execution_failure(self):
        """测试步骤执行失败"""

        def failing_function(data):
            raise ValueError("Processing failed")

        pipeline = Pipeline([1, 2, 3])

        with pytest.raises(
            RuntimeError, match="Pipeline failed at step 'failing_function'"
        ):
            pipeline.then(failing_function)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].success is False
        assert isinstance(pipeline.steps[0].error, ValueError)

    def test_multiout_step_wrong_return_count(self):
        """测试multiout_step返回值数量错误"""

        @multiout_step(stats="statistics", metadata="info")  # 期望3个值
        def wrong_return_count(data):
            return data, {"count": len(data)}  # 只返回2个值

        pipeline = Pipeline([1, 2, 3])

        with pytest.raises(RuntimeError, match="expected 3 return values but got 2"):
            pipeline.then(wrong_return_count)

    def test_no_data_error(self):
        """测试没有数据时的错误"""
        pipeline = Pipeline()

        def dummy_func(x):
            return x

        with pytest.raises(ValueError, match="No data to process"):
            pipeline.then(dummy_func)

    def test_undecorated_function_warning(self):
        """测试未装饰的函数会产生警告"""

        def undecorated_func(data):
            return [x * 2 for x in data]

        pipeline = Pipeline([1, 2, 3])

        with pytest.warns(UserWarning, match="is not decorated with @pipeline_step"):
            pipeline.then(undecorated_func)

        assert pipeline.data == [2, 4, 6]  # 功能仍然正常
