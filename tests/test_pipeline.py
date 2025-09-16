import pytest
import tempfile
from pathlib import Path
import pickle
import warnings

from tidymut.core.pipeline import (
    Pipeline,
    create_pipeline,
    pipeline_step,
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

        @pipeline_step(name="process_data")
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

        @pipeline_step(name="step1")
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

        @pipeline_step(name="process_data")
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

        @pipeline_step(name="normalize")
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


class TestDelayedPipeline:
    """测试Pipeline的延迟执行功能"""

    def test_delayed_then_does_not_execute_immediately(self):
        """测试delayed_then不会立即执行步骤"""
        pipeline = Pipeline([1, 2, 3])

        def double(data):
            return [x * 2 for x in data]

        # 添加延迟步骤，数据不应该改变
        pipeline.delayed_then(double)

        assert pipeline.data == [1, 2, 3]  # 数据未改变
        assert len(pipeline.delayed_steps) == 1  # 有一个延迟步骤
        assert len(pipeline.steps) == 0  # 没有已执行的步骤
        assert pipeline.has_pending_steps is True

    def test_execute_runs_delayed_steps(self):
        """测试execute方法执行延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def double(data):
            return [x * 2 for x in data]

        def add_one(data):
            return [x + 1 for x in data]

        # 添加延迟步骤
        pipeline.delayed_then(double).delayed_then(add_one)

        assert pipeline.data == [1, 2, 3]  # 执行前数据未变
        assert len(pipeline.delayed_steps) == 2

        # 执行所有延迟步骤
        result = pipeline.execute()

        assert result is pipeline  # 返回self用于链式调用
        assert pipeline.data == [3, 5, 7]  # (1*2)+1, (2*2)+1, (3*2)+1
        assert len(pipeline.delayed_steps) == 0  # 延迟步骤已清空
        assert len(pipeline.steps) == 2  # 现在有2个已执行步骤
        assert pipeline.has_pending_steps is False

    def test_execute_with_step_count(self):
        """测试execute方法执行指定数量的步骤"""
        pipeline = Pipeline([1, 2, 3])

        def double(data):
            return [x * 2 for x in data]

        def add_one(data):
            return [x + 1 for x in data]

        def multiply_by_three(data):
            return [x * 3 for x in data]

        # 添加3个延迟步骤
        pipeline.delayed_then(double).delayed_then(add_one).delayed_then(
            multiply_by_three
        )

        # 只执行前2个步骤
        pipeline.execute(2)

        assert pipeline.data == [3, 5, 7]  # 只执行了double和add_one
        assert len(pipeline.delayed_steps) == 1  # 还有1个延迟步骤
        assert len(pipeline.steps) == 2  # 执行了2个步骤

        # 执行剩余步骤
        pipeline.execute()

        assert pipeline.data == [9, 15, 21]  # 最后执行multiply_by_three
        assert len(pipeline.delayed_steps) == 0

    def test_execute_with_specific_indices(self):
        """测试execute方法执行指定索引的步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x + 10 for x in data]

        def step2(data):
            return [x * 2 for x in data]

        def step3(data):
            return [x - 5 for x in data]

        # 添加3个延迟步骤
        pipeline.delayed_then(step1).delayed_then(step2).delayed_then(step3)

        # 执行第0和第2个步骤（跳过第1个）
        pipeline.execute([0, 2])

        assert len(pipeline.delayed_steps) == 1  # 还有1个延迟步骤（step2）
        assert len(pipeline.steps) == 2  # 执行了2个步骤

        # 验证执行顺序：先执行step1，再执行step3
        # [1,2,3] -> step1 -> [11,12,13] -> step3 -> [6,7,8]
        assert pipeline.data == [6, 7, 8]

    def test_execute_all(self):
        """测试execute_all"""
        pipeline = Pipeline([1, 2, 3])

        def double(data):
            return [x * 2 for x in data]

        pipeline.delayed_then(double)
        pipeline.execute()

        assert pipeline.data == [2, 4, 6]
        assert len(pipeline.delayed_steps) == 0

    def test_mixed_usage_warning(self):
        """测试混用delayed_then和then时的警告"""
        pipeline = Pipeline([1, 2, 3])

        @pipeline_step
        def double(data):
            return [x * 2 for x in data]

        @pipeline_step
        def add_one(data):
            return [x + 1 for x in data]

        # 添加延迟步骤
        pipeline.delayed_then(double)

        # 使用then应该产生警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline.then(add_one)

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "pending delayed steps" in str(w[0].message)

        # 验证执行结果：then立即执行，基于原始数据
        assert pipeline.data == [2, 3, 4]  # [1,2,3] + 1
        assert len(pipeline.delayed_steps) == 1  # 延迟步骤仍然存在

    def test_has_pending_steps_property(self):
        """测试has_pending_steps属性"""
        pipeline = Pipeline([1, 2, 3])

        # 初始状态
        assert pipeline.has_pending_steps is False

        # 添加延迟步骤
        pipeline.delayed_then(lambda x: x)
        assert pipeline.has_pending_steps is True

        # 执行延迟步骤
        pipeline.execute()
        assert pipeline.has_pending_steps is False

        # 清空延迟步骤
        pipeline.delayed_then(lambda x: x)
        assert pipeline.has_pending_steps is True

    def test_get_delayed_steps_info(self):
        """测试获取延迟步骤信息"""
        pipeline = Pipeline([1, 2, 3])

        @pipeline_step
        def decorated_step(data):
            return data

        def regular_step(data):
            return data

        # 添加不同类型的步骤
        pipeline.delayed_then(decorated_step)
        pipeline.delayed_then(regular_step)

        info = pipeline.get_delayed_steps_info()

        assert len(info) == 2

        # 检查第一个步骤（装饰过的）
        assert info[0]["index"] == 0
        assert info[0]["name"] == "decorated_step"
        assert info[0]["is_pipeline_step"] is True
        assert info[0]["step_type"] == "single_output"

        # 检查第二个步骤（普通函数）
        assert info[1]["index"] == 1
        assert info[1]["name"] == "regular_step"
        assert info[1]["is_pipeline_step"] is False
        assert info[1]["step_type"] == "unknown"

    def test_visualize_pipeline_with_delayed_steps(self):
        """测试包含延迟步骤的pipeline可视化"""
        pipeline = Pipeline([1, 2, 3], name="TestPipeline")

        def executed_step(data):
            return [x * 2 for x in data]

        @pipeline_step
        def delayed_step(data):
            """This is a delayed step"""
            return data

        # 执行一个步骤，添加一个延迟步骤
        pipeline.then(executed_step)
        pipeline.delayed_then(delayed_step)

        visualization = pipeline.visualize_pipeline()

        # 检查包含的关键信息
        assert "TestPipeline" in visualization
        assert "executed_step" in visualization
        assert "Delayed Steps:" in visualization
        assert "delayed_step" in visualization
        assert "This is a delayed step" in visualization
        assert "Delayed steps: 1" in visualization
        assert "✓" in visualization  # 执行成功的标记
        assert "⏸" in visualization  # 延迟步骤的标记

    def test_execution_summary_with_delayed_steps(self):
        """测试包含延迟步骤的执行摘要"""
        pipeline = Pipeline([1, 2, 3], name="TestPipeline")

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        # 执行一个步骤，添加一个延迟步骤
        pipeline.then(step1)
        pipeline.delayed_then(step2)

        summary = pipeline.get_execution_summary()

        assert summary["pipeline_name"] == "TestPipeline"
        assert summary["total_steps"] == 1  # 已执行步骤
        assert summary["delayed_steps"] == 1  # 延迟步骤
        assert summary["successful_steps"] == 1
        assert summary["failed_steps"] == 0
        assert len(summary["delayed_steps_info"]) == 1
        assert summary["delayed_steps_info"][0]["name"] == "step2"

    def test_copy_preserves_delayed_steps(self):
        """测试copy方法保留延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        # 添加延迟步骤
        pipeline.delayed_then(step1).delayed_then(step2)

        # 复制pipeline
        copied = pipeline.copy()

        assert len(copied.delayed_steps) == 2
        assert copied.has_pending_steps is True
        assert copied.data == [1, 2, 3]

        # 原pipeline和复制的pipeline应该独立
        pipeline.execute()
        assert pipeline.data == [3, 5, 7]
        assert copied.data == [1, 2, 3]  # 复制的pipeline数据未变

    def test_str_and_repr_with_delayed_steps(self):
        """测试包含延迟步骤的字符串表示"""
        pipeline = Pipeline([1, 2, 3], name="TestPipeline")

        def step1(data):
            return data

        # 执行一个步骤，添加延迟步骤
        pipeline.then(step1)
        pipeline.delayed_then(step1)

        str_repr = str(pipeline)
        repr_str = repr(pipeline)

        assert "TestPipeline" in str_repr
        assert "1/1 steps executed" in str_repr
        assert "1 delayed" in str_repr

        assert "TestPipeline" in repr_str
        assert "steps=1" in repr_str
        assert "delayed=1" in repr_str

    def test_pipeline_step_decorator_with_delayed_execution(self):
        """测试装饰器函数在延迟执行中的行为"""
        pipeline = Pipeline([1, 2, 3])

        @pipeline_step
        def double_values(data):
            """Double all values in the list"""
            return [x * 2 for x in data]

        @multiout_step(stats="statistics")
        def analyze_data(data):
            """Analyze data and return stats"""
            total = sum(data)
            return data, {"sum": total, "count": len(data)}

        # 使用延迟执行
        pipeline.delayed_then(double_values)
        pipeline.delayed_then(analyze_data)

        # 执行延迟步骤
        pipeline.execute()

        assert pipeline.data == [2, 4, 6]
        assert "analyze_data.statistics" in pipeline.artifacts
        assert pipeline.artifacts["analyze_data.statistics"]["sum"] == 12
        assert pipeline.artifacts["analyze_data.statistics"]["count"] == 3

    def test_error_handling_in_delayed_execution(self):
        """测试延迟执行中的错误处理"""
        pipeline = Pipeline([1, 2, 3])

        def working_step(data):
            return [x * 2 for x in data]

        def failing_step(data):
            raise ValueError("Intentional error")

        # 添加正常步骤和失败步骤
        pipeline.delayed_then(working_step)
        pipeline.delayed_then(failing_step)

        # 执行应该在第二步失败
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.execute()

        assert "Pipeline failed at delayed step 'failing_step'" in str(exc_info.value)

        # 第一步应该已执行成功
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].success is True
        assert pipeline.steps[1].success is False
        assert pipeline.data == [2, 4, 6]  # 第一步的结果

    def test_add_delayed_step_at_end(self):
        """测试在末尾添加延迟步骤（默认行为）"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        def step3(data):
            return [x * 3 for x in data]

        # 先添加两个步骤
        pipeline.delayed_then(step1).delayed_then(step2)

        # 在末尾添加第三个步骤
        pipeline.add_delayed_step(step3)

        assert len(pipeline.delayed_steps) == 3
        assert pipeline.delayed_steps[0].name == "step1"
        assert pipeline.delayed_steps[1].name == "step2"
        assert pipeline.delayed_steps[2].name == "step3"

        # 执行验证顺序
        pipeline.execute()
        # [1,2,3] -> *2 -> [2,4,6] -> +1 -> [3,5,7] -> *3 -> [9,15,21]
        assert pipeline.data == [9, 15, 21]

    def test_add_delayed_step_at_beginning(self):
        """测试在开头插入延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        def step0(data):  # 要插入到开头的步骤
            return [x + 10 for x in data]

        # 先添加两个步骤
        pipeline.delayed_then(step1).delayed_then(step2)

        # 在开头插入步骤
        pipeline.add_delayed_step(step0, 0)

        assert len(pipeline.delayed_steps) == 3
        assert pipeline.delayed_steps[0].name == "step0"
        assert pipeline.delayed_steps[1].name == "step1"
        assert pipeline.delayed_steps[2].name == "step2"

        # 执行验证顺序
        pipeline.execute()
        # [1,2,3] -> +10 -> [11,12,13] -> *2 -> [22,24,26] -> +1 -> [23,25,27]
        assert pipeline.data == [23, 25, 27]

    def test_add_delayed_step_at_middle(self):
        """测试在中间位置插入延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        def step_middle(data):
            return [x + 100 for x in data]

        # 先添加两个步骤
        pipeline.delayed_then(step1).delayed_then(step2)

        # 在中间插入步骤
        pipeline.add_delayed_step(step_middle, 1)

        assert len(pipeline.delayed_steps) == 3
        assert pipeline.delayed_steps[0].name == "step1"
        assert pipeline.delayed_steps[1].name == "step_middle"
        assert pipeline.delayed_steps[2].name == "step2"

        # 执行验证顺序
        pipeline.execute()
        # [1,2,3] -> *2 -> [2,4,6] -> +100 -> [102,104,106] -> +1 -> [103,105,107]
        assert pipeline.data == [103, 105, 107]

    def test_add_delayed_step_negative_index(self):
        """测试使用负索引插入延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return [x * 2 for x in data]

        def step2(data):
            return [x + 1 for x in data]

        def step_before_last(data):
            return [x + 100 for x in data]

        # 先添加两个步骤
        pipeline.delayed_then(step1).delayed_then(step2)

        # 在倒数第一个位置插入（即在step2之前）
        pipeline.add_delayed_step(step_before_last, -1)

        assert len(pipeline.delayed_steps) == 3
        assert pipeline.delayed_steps[0].name == "step1"
        assert pipeline.delayed_steps[1].name == "step_before_last"
        assert pipeline.delayed_steps[2].name == "step2"

    def test_add_delayed_step_out_of_bounds(self):
        """测试超出范围的索引会被自动调整"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return data

        def step2(data):
            return data

        # 先添加一个步骤
        pipeline.delayed_then(step1)

        # 尝试在超出范围的位置插入，应该自动调整到末尾
        pipeline.add_delayed_step(step2, 100)

        assert len(pipeline.delayed_steps) == 2
        assert pipeline.delayed_steps[1].name == "step2"

    def test_remove_delayed_step_by_index(self):
        """测试通过索引删除延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return data

        def step2(data):
            return data

        def step3(data):
            return data

        # 添加三个步骤
        pipeline.delayed_then(step1).delayed_then(step2).delayed_then(step3)

        # 删除中间的步骤
        pipeline.remove_delayed_step(1)

        assert len(pipeline.delayed_steps) == 2
        assert pipeline.delayed_steps[0].name == "step1"
        assert pipeline.delayed_steps[1].name == "step3"

    def test_remove_delayed_step_by_name(self):
        """测试通过名称删除延迟步骤"""
        pipeline = Pipeline([1, 2, 3])

        def step1(data):
            return data

        @pipeline_step(name="custom_name")
        def step2(data):
            return data

        def step3(data):
            return data

        # 添加三个步骤
        pipeline.delayed_then(step1).delayed_then(step2).delayed_then(step3)

        # 通过名称删除步骤
        pipeline.remove_delayed_step("custom_name")

        assert len(pipeline.delayed_steps) == 2
        assert pipeline.delayed_steps[0].name == "step1"
        assert pipeline.delayed_steps[1].name == "step3"

    def test_remove_delayed_step_invalid_cases(self):
        """测试删除延迟步骤的无效情况"""
        pipeline = Pipeline([1, 2, 3])

        @pipeline_step
        def step1(data):
            return data

        pipeline.delayed_then(step1)

        # 测试不存在的名称
        with pytest.raises(ValueError):
            pipeline.remove_delayed_step("nonexistent")

        # 测试无效类型
        with pytest.raises(TypeError):
            pipeline.remove_delayed_step(1.5)  # type: ignore

    def test_delayed_step_management_integration(self):
        """测试延迟步骤管理功能的集成使用"""
        pipeline = Pipeline([1, 2, 3])

        def multiply(data):
            return [x * 2 for x in data]

        def add(data):
            return [x + 10 for x in data]

        def subtract(data):
            return [x - 5 for x in data]

        def divide(data):
            return [x // 2 for x in data]

        # 构建复杂的延迟执行序列
        pipeline.delayed_then(multiply)  # 0: multiply
        pipeline.add_delayed_step(add, 0)  # 在开头插入add: [add, multiply]
        pipeline.delayed_then(subtract)  # 末尾添加subtract: [add, multiply, subtract]
        pipeline.add_delayed_step(
            divide, 2
        )  # 在位置2插入divide: [add, multiply, divide, subtract]
        # 删除add
        pipeline.remove_delayed_step("add")  # [multiply, divide, subtract]
        # 验证最终的执行顺序
        assert len(pipeline.delayed_steps) == 3
        assert pipeline.delayed_steps[0].name == "multiply"
        assert pipeline.delayed_steps[1].name == "divide"
        assert pipeline.delayed_steps[2].name == "subtract"

        # 执行并验证结果
        pipeline.execute()
        # [1,2,3] -> -5 -> [-4,-3,-2] -> *2 -> [-8,-6,-4] -> //2 -> [-4,-3,-2]
        assert pipeline.data == [-4, -3, -2]

    def test_empty_delayed_steps_execute(self):
        """测试在没有延迟步骤时调用execute"""
        pipeline = Pipeline([1, 2, 3])

        # 没有延迟步骤时调用execute
        result = pipeline.execute()

        assert result is pipeline
        assert pipeline.data == [1, 2, 3]  # 数据未改变
        assert len(pipeline.steps) == 0
        assert len(pipeline.delayed_steps) == 0
