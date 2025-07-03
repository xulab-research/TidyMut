import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from tidymut.core.dataset import MutationDataset
from tidymut.core.mutation import (
    AminoAcidMutation,
    CodonMutation,
    AminoAcidMutationSet,
    CodonMutationSet,
)
from tidymut.core.sequence import (
    ProteinSequence,
    DNASequence,
    RNASequence,
)


class TestMutationDatasetInit:
    """测试MutationDataset初始化"""

    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = MutationDataset()

        assert dataset.name is None
        assert len(dataset.reference_sequences) == 0
        assert len(dataset.mutation_sets) == 0
        assert len(dataset) == 0

    def test_dataset_creation_with_name(self):
        """测试带名称的数据集创建"""
        dataset = MutationDataset(name="test_dataset")

        assert dataset.name == "test_dataset"
        assert len(dataset) == 0


class TestReferenceSequenceManagement:
    """测试参考序列管理功能"""

    def test_add_reference_sequence(self):
        """测试添加参考序列"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", name="test_protein")

        dataset.add_reference_sequence("ref1", protein_seq)

        assert "ref1" in dataset.reference_sequences
        assert dataset.reference_sequences["ref1"] == protein_seq
        assert dataset.list_reference_sequences() == ["ref1"]

    def test_add_duplicate_reference_sequence(self):
        """测试添加重复的参考序列ID"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")

        dataset.add_reference_sequence("ref1", protein_seq)

        with pytest.raises(
            ValueError, match="Reference sequence with ID 'ref1' already exists"
        ):
            dataset.add_reference_sequence("ref1", protein_seq)

    def test_get_reference_sequence(self):
        """测试获取参考序列"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        retrieved_seq = dataset.get_reference_sequence("ref1")
        assert retrieved_seq == protein_seq

    def test_get_nonexistent_reference_sequence(self):
        """测试获取不存在的参考序列"""
        dataset = MutationDataset()

        with pytest.raises(
            ValueError, match="Reference sequence with ID 'nonexistent' not found"
        ):
            dataset.get_reference_sequence("nonexistent")

    def test_remove_reference_sequence(self):
        """测试删除参考序列"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        dataset.remove_reference_sequence("ref1")

        assert "ref1" not in dataset.reference_sequences
        assert len(dataset.list_reference_sequences()) == 0

    def test_remove_referenced_sequence(self):
        """测试删除被突变集引用的参考序列"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加引用该序列的突变集
        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation])
        dataset.add_mutation_set(mutation_set, "ref1")

        with pytest.raises(
            ValueError, match="Cannot remove sequence 'ref1' as it is referenced"
        ):
            dataset.remove_reference_sequence("ref1")


class TestMutationSetManagement:
    """测试突变集管理功能"""

    def test_add_mutation_set(self):
        """测试添加突变集"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation], name="test_set")

        dataset.add_mutation_set(mutation_set, "ref1", label=1.0)

        assert len(dataset) == 1
        assert dataset.mutation_sets[0] == mutation_set
        assert dataset.get_mutation_set_reference(0) == "ref1"
        assert dataset.get_mutation_set_label(0) == 1.0

    def test_add_mutation_set_without_reference(self):
        """测试在没有参考序列的情况下添加突变集"""
        dataset = MutationDataset()

        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation])

        with pytest.raises(
            ValueError, match="Reference sequence with ID 'ref1' not found"
        ):
            dataset.add_mutation_set(mutation_set, "ref1")

    def test_add_multiple_mutation_sets(self):
        """测试批量添加突变集"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        mutation_sets = [
            AminoAcidMutationSet([AminoAcidMutation("A", 0, "V")]),
            AminoAcidMutationSet([AminoAcidMutation("C", 1, "G")]),
            AminoAcidMutationSet([AminoAcidMutation("D", 2, "E")]),
        ]
        reference_ids = ["ref1", "ref1", "ref1"]
        labels = [1.0, 2.0, 3.0]

        dataset.add_mutation_sets(mutation_sets, reference_ids, labels)

        assert len(dataset) == 3
        for i in range(3):
            assert dataset.get_mutation_set_label(i) == labels[i]

    def test_add_mutation_sets_mismatched_lengths(self):
        """测试添加突变集时长度不匹配"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        mutation_sets = [AminoAcidMutationSet([AminoAcidMutation("A", 0, "V")])]
        reference_ids = ["ref1", "ref1"]  # 长度不匹配

        with pytest.raises(ValueError, match="Number of reference_ids must match"):
            dataset.add_mutation_sets(mutation_sets, reference_ids)

    def test_set_mutation_set_reference(self):
        """测试设置突变集的参考序列"""
        dataset = MutationDataset()
        protein_seq1 = ProteinSequence("ACDEFG", name="test_protein1")
        protein_seq2 = ProteinSequence("GHIKLM", name="test_protein2")
        dataset.add_reference_sequence("ref1", protein_seq1)
        dataset.add_reference_sequence("ref2", protein_seq2)

        mutation_set = AminoAcidMutationSet([AminoAcidMutation("A", 0, "V")])
        dataset.add_mutation_set(mutation_set, "ref1")

        # 更改参考序列
        dataset.set_mutation_set_reference(0, "ref2")
        assert dataset.get_mutation_set_reference(0) == "ref2"

    def test_remove_mutation_set(self):
        """测试删除突变集"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加三个突变集
        for i in range(3):
            mutation_set = AminoAcidMutationSet(
                [AminoAcidMutation("A", 0, f"{chr(82+i)}")]
            )
            dataset.add_mutation_set(mutation_set, "ref1", label=i)

        # 删除中间的突变集
        dataset.remove_mutation_set(1)

        assert len(dataset) == 2
        assert dataset.get_mutation_set_label(0) == 0
        assert dataset.get_mutation_set_label(1) == 2  # 索引已更新


class TestValidation:
    """测试验证功能"""

    def test_validate_against_references_valid(self):
        """测试有效突变的验证"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加有效突变
        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation], name="valid_set")
        dataset.add_mutation_set(mutation_set, "ref1")

        results = dataset.validate_against_references()

        assert len(results["valid_mutation_sets"]) == 1
        assert len(results["invalid_mutation_sets"]) == 0
        assert len(results["position_mismatches"]) == 0

    def test_validate_against_references_position_mismatch(self):
        """测试位置不匹配的验证"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加野生型不匹配的突变
        mutation = AminoAcidMutation("V", 0, "L")  # 位置0应该是A而不是V
        mutation_set = AminoAcidMutationSet([mutation], name="mismatch_set")
        dataset.add_mutation_set(mutation_set, "ref1")

        results = dataset.validate_against_references()

        assert len(results["position_mismatches"]) == 1
        assert results["position_mismatches"][0]["expected"] == "A"
        assert results["position_mismatches"][0]["found"] == "V"

    def test_validate_against_references_out_of_bounds(self):
        """测试越界位置的验证"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加越界突变
        mutation = AminoAcidMutation("A", 10, "V")  # 序列长度只有6
        mutation_set = AminoAcidMutationSet([mutation], name="out_of_bounds_set")
        dataset.add_mutation_set(mutation_set, "ref1")

        results = dataset.validate_against_references()

        assert len(results["invalid_mutation_sets"]) == 1
        assert "exceeds sequence length" in results["invalid_mutation_sets"][0]["error"]


class TestDataConversion:
    """测试数据转换功能"""

    def test_to_dataframe(self):
        """测试转换为DataFrame"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加突变集
        mutations = [AminoAcidMutation("A", 0, "V"), AminoAcidMutation("C", 1, "G")]
        mutation_set = AminoAcidMutationSet(mutations, name="test_set")
        dataset.add_mutation_set(mutation_set, "ref1", label=1.5)

        df = dataset.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 两个突变
        assert df.iloc[0]["mutation_set_name"] == "test_set"
        assert df.iloc[0]["reference_id"] == "ref1"
        assert df.iloc[0]["label"] == 1.5
        assert df.iloc[0]["mutation_string"] == "A0V"
        assert df.iloc[1]["mutation_string"] == "C1G"

    def test_convert_codon_to_amino_acid_sets(self):
        """测试密码子突变集转换为氨基酸突变集"""
        dataset = MutationDataset()
        dna_seq = DNASequence("ATGGCCTAT", name="test_dna")
        dataset.add_reference_sequence("ref1", dna_seq)

        # 添加密码子突变集
        codon_mutation = CodonMutation("ATG", 0, "CTG")  # M -> L
        codon_set = CodonMutationSet([codon_mutation])
        dataset.add_mutation_set(codon_set, "ref1", label=2.0)

        # 转换
        converted_dataset = dataset.convert_codon_to_amino_acid_sets(
            convert_labels=True
        )

        assert len(converted_dataset) == 1
        converted_set = converted_dataset.mutation_sets[0]
        assert isinstance(converted_set, AminoAcidMutationSet)
        assert len(converted_set) == 1
        assert str(converted_set.mutations[0]) == "M0L"
        assert converted_dataset.get_mutation_set_label(0) == 2.0


class TestFiltering:
    """测试过滤功能"""

    def test_filter_by_reference(self):
        """测试按参考序列过滤"""
        dataset = MutationDataset()

        # 添加两个参考序列
        protein_seq1 = ProteinSequence("ACDEFG", name="protein1")
        protein_seq2 = ProteinSequence("GHIKLM", name="protein2")
        dataset.add_reference_sequence("ref1", protein_seq1)
        dataset.add_reference_sequence("ref2", protein_seq2)

        # 为每个参考序列添加突变集
        for i in range(3):
            mutation_set = AminoAcidMutationSet([AminoAcidMutation("A", 0, "V")])
            dataset.add_mutation_set(mutation_set, "ref1")

        for i in range(2):
            mutation_set = AminoAcidMutationSet([AminoAcidMutation("G", 0, "A")])
            dataset.add_mutation_set(mutation_set, "ref2")

        # 过滤
        filtered = dataset.filter_by_reference("ref1")

        assert len(filtered) == 3
        assert "ref1" in filtered.reference_sequences
        assert "ref2" not in filtered.reference_sequences

    def test_filter_by_mutation_type(self):
        """测试按突变类型过滤"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加氨基酸突变
        aa_mutation = AminoAcidMutation("A", 0, "V")
        aa_set = AminoAcidMutationSet([aa_mutation])
        dataset.add_mutation_set(aa_set, "ref1")

        # 添加DNA序列和密码子突变
        dna_seq = DNASequence("ATGGCCTAT", name="test_dna")
        dataset.add_reference_sequence("ref2", dna_seq)
        codon_mutation = CodonMutation("ATG", 0, "CTG")
        codon_set = CodonMutationSet([codon_mutation])
        dataset.add_mutation_set(codon_set, "ref2")

        # 过滤氨基酸突变
        filtered = dataset.filter_by_mutation_type(AminoAcidMutation)

        assert len(filtered) == 1
        assert isinstance(filtered.mutation_sets[0], AminoAcidMutationSet)

    def test_filter_by_effect_type(self):
        """测试按效果类型过滤"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加不同效果类型的突变
        mutations = [
            AminoAcidMutation("A", 0, "V"),  # missense
            AminoAcidMutation("C", 1, "C"),  # synonymous
            AminoAcidMutation("D", 2, "*"),  # nonsense
        ]

        for mut in mutations:
            mutation_set = AminoAcidMutationSet([mut])
            dataset.add_mutation_set(mutation_set, "ref1")

        # 过滤missense突变
        filtered = dataset.filter_by_effect_type("missense")

        assert len(filtered) == 1
        assert filtered.mutation_sets[0].mutations[0].is_missense()


class TestStatistics:
    """测试统计功能"""

    def test_get_statistics(self):
        """测试获取统计信息"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加单突变和多突变集
        single_mutation = AminoAcidMutationSet([AminoAcidMutation("A", 0, "V")])
        dataset.add_mutation_set(single_mutation, "ref1")

        multiple_mutations = AminoAcidMutationSet(
            [AminoAcidMutation("A", 0, "V"), AminoAcidMutation("C", 1, "G")]
        )
        dataset.add_mutation_set(multiple_mutations, "ref1")

        stats = dataset.get_statistics()

        assert stats["total_mutation_sets"] == 2
        assert stats["total_mutations"] == 3
        assert stats["single_mutation_sets"] == 1
        assert stats["multiple_mutation_sets"] == 1
        assert stats["average_mutations_per_set"] == 1.5

    def test_get_position_coverage(self):
        """测试获取位置覆盖统计"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加覆盖不同位置的突变
        positions = [0, 1, 3, 5]  # 跳过位置2和4
        for pos in positions:
            mutation = AminoAcidMutation(protein_seq.get_residue(pos), pos, "A")
            mutation_set = AminoAcidMutationSet([mutation])
            dataset.add_mutation_set(mutation_set, "ref1")

        coverage = dataset.get_position_coverage("ref1")

        assert coverage["sequence_length"] == 6
        assert coverage["covered_positions"] == 4
        assert coverage["uncovered_positions"] == 2
        assert coverage["coverage_percentage"] == pytest.approx(66.67, rel=0.01)
        assert coverage["position_list"] == [0, 1, 3, 5]


class TestSaveLoad:
    """测试保存和加载功能"""

    def test_save_load_pickle(self):
        """测试pickle格式保存和加载"""
        # 创建数据集
        dataset = MutationDataset(name="test_dataset")
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation], name="test_set")
        dataset.add_mutation_set(mutation_set, "ref1", label=1.0)

        # 保存和加载
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            dataset.save(tmp.name, save_type="pickle")
            loaded_dataset = MutationDataset.load(tmp.name)

        # 验证
        assert loaded_dataset.name == "test_dataset"
        assert len(loaded_dataset) == 1
        assert loaded_dataset.get_mutation_set_label(0) == 1.0

        # 清理
        Path(tmp.name).unlink()

    def test_save_load_dataframe(self):
        """测试dataframe格式保存和加载"""
        # 创建数据集
        dataset = MutationDataset(name="test_dataset")
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        mutation = AminoAcidMutation("A", 0, "V")
        mutation_set = AminoAcidMutationSet([mutation], name="test_set")
        dataset.add_mutation_set(mutation_set, "ref1", label=2.5)
        dataset.metadata["test_key"] = "test_value"

        # 保存和加载
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_data"
            dataset.save(str(base_path), save_type="dataframe")
            loaded_dataset = MutationDataset.load(str(base_path), load_type="dataframe")

        # 验证
        assert loaded_dataset.name == "test_dataset"
        assert len(loaded_dataset) == 1
        assert loaded_dataset.get_mutation_set_label(0) == 2.5
        assert loaded_dataset.metadata["test_key"] == "test_value"

    def test_save_load_by_reference(self):
        """测试按参考序列格式保存和加载"""
        # 创建数据集
        dataset = MutationDataset(name="test_dataset")

        # 添加两个参考序列
        protein_seq1 = ProteinSequence("ACDEFG", name="protein1")
        protein_seq2 = ProteinSequence("GHIKLM", name="protein2")
        dataset.add_reference_sequence("ref1", protein_seq1)
        dataset.add_reference_sequence("ref2", protein_seq2)

        # 为第一个参考序列添加多个突变集
        mutation1 = AminoAcidMutation("A", 0, "V")
        mutation_set1 = AminoAcidMutationSet([mutation1])
        dataset.add_mutation_set(mutation_set1, "ref1", label=1.0)

        # 添加多突变集合
        mutation1_multi = AminoAcidMutation("A", 0, "V")
        mutation2_multi = AminoAcidMutation("C", 1, "G")
        mutation_set1_multi = AminoAcidMutationSet([mutation1_multi, mutation2_multi])
        dataset.add_mutation_set(mutation_set1_multi, "ref1", label=3.0)

        # 为第二个参考序列添加突变
        mutation2 = AminoAcidMutation("G", 0, "A")
        mutation_set2 = AminoAcidMutationSet([mutation2])
        dataset.add_mutation_set(mutation_set2, "ref2", label=2.0)

        # 添加无标签的突变集
        mutation3 = AminoAcidMutation("H", 1, "R")
        mutation_set3 = AminoAcidMutationSet([mutation3])
        dataset.add_mutation_set(mutation_set3, "ref2")  # 无标签

        # 保存和加载
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save_by_reference(tmpdir)

            # 验证保存的文件结构
            base_path = Path(tmpdir)
            ref_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            assert (
                len(ref_dirs) == 2
            ), f"Expected 2 reference directories, got {len(ref_dirs)}"

            # 验证每个目录包含必需文件
            for ref_dir in ref_dirs:
                assert (
                    ref_dir / "data.csv"
                ).exists(), f"Missing data.csv in {ref_dir.name}"
                assert (
                    ref_dir / "wt.fasta"
                ).exists(), f"Missing wt.fasta in {ref_dir.name}"
                assert (
                    ref_dir / "metadata.json"
                ).exists(), f"Missing metadata.json in {ref_dir.name}"

                # 验证metadata.json格式
                with open(ref_dir / "metadata.json") as f:
                    metadata = json.load(f)
                    required_fields = [
                        "reference_id",
                        "sequence_name",
                        "sequence_type",
                        "sequence_length",
                        "num_mutation_sets",
                        "total_mutations",
                        "covered_positions",
                        "coverage_percentage",
                        "num_unique_labels",
                        "has_unlabeled",
                        "dataset_name",
                    ]
                    for field in required_fields:
                        assert field in metadata, f"Missing field {field} in metadata"

            # 加载数据集
            loaded_dataset = MutationDataset.load_by_reference(tmpdir, "loaded_dataset")

        # 验证基本属性
        assert loaded_dataset.name == "loaded_dataset"
        assert (
            len(loaded_dataset) == 4
        ), f"Expected 4 mutation sets, got {len(loaded_dataset)}"
        assert len(loaded_dataset.reference_sequences) == 2
        assert "ref1" in loaded_dataset.reference_sequences
        assert "ref2" in loaded_dataset.reference_sequences

        # 验证参考序列内容
        loaded_seq1 = loaded_dataset.reference_sequences["ref1"]
        loaded_seq2 = loaded_dataset.reference_sequences["ref2"]
        assert str(loaded_seq1) == "ACDEFG"
        assert str(loaded_seq2) == "GHIKLM"
        assert loaded_seq1.name == "protein1"
        assert loaded_seq2.name == "protein2"

        # 验证突变集和标签
        ref1_sets = []
        ref2_sets = []

        for i, (mutation_set, ref_id) in enumerate(loaded_dataset):
            label = loaded_dataset.mutation_set_labels.get(i, "")

            if ref_id == "ref1":
                ref1_sets.append((mutation_set, label))
            elif ref_id == "ref2":
                ref2_sets.append((mutation_set, label))

        # 验证ref1的突变集
        assert (
            len(ref1_sets) == 2
        ), f"Expected 2 mutation sets for ref1, got {len(ref1_sets)}"

        # 查找单突变和多突变
        single_mut = None
        multi_mut = None
        for mut_set, label in ref1_sets:
            if len(mut_set) == 1:
                single_mut = (mut_set, label)
            elif len(mut_set) == 2:
                multi_mut = (mut_set, label)

        assert single_mut is not None, "Single mutation not found for ref1"
        assert multi_mut is not None, "Multi mutation not found for ref1"
        assert single_mut[1] == 1.0, f"Expected label 1.0, got {single_mut[1]}"
        assert multi_mut[1] == 3.0, f"Expected label '3.0', got {multi_mut[1]}"

        # 验证ref2的突变集
        assert (
            len(ref2_sets) == 2
        ), f"Expected 2 mutation sets for ref2, got {len(ref2_sets)}"

        # 检查标签
        labels = [label for _, label in ref2_sets]
        assert 2.0 in labels, "Label 2.0 not found in ref2"
        assert pd.isna(labels).any(), "Empty label (unlabeled) not found in ref2"

        # 验证统计信息
        stats = loaded_dataset.get_statistics()
        assert stats["total_mutation_sets"] == 4
        assert stats["total_mutations"] == 5  # 1 + 2 + 1 + 1
        assert stats["num_reference_sequences"] == 2

        print("All tests passed!")

    def test_save_load_error_handling(self):
        """测试错误处理"""
        # 测试空目录
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                MutationDataset.load_by_reference(tmpdir)
                assert False, "Should raise ValueError for empty directory"
            except ValueError as e:
                assert "No reference directories found" in str(e)

        # 测试不存在的目录
        try:
            MutationDataset.load_by_reference("/nonexistent/path")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass

        print("Error handling tests passed!")

    def test_sanitize_filename(self):
        """测试文件名清理功能"""
        # 测试包含特殊字符的文件名
        assert MutationDataset._sanitize_filename("test:name") == "test_name"
        assert MutationDataset._sanitize_filename("test/name") == "test_name"
        assert MutationDataset._sanitize_filename("test<>name") == "test__name"

        # 测试空字符串
        assert MutationDataset._sanitize_filename("") == "unnamed"

        # 测试过长文件名
        long_name = "a" * 300
        sanitized = MutationDataset._sanitize_filename(long_name)
        assert len(sanitized) == 200


class TestSpecialMethods:
    """测试特殊方法"""

    def test_len(self):
        """测试__len__方法"""
        dataset = MutationDataset()
        assert len(dataset) == 0

        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        for i in range(5):
            mutation = AminoAcidMutation("A", 0, "V")
            mutation_set = AminoAcidMutationSet([mutation])
            dataset.add_mutation_set(mutation_set, "ref1")

        assert len(dataset) == 5

    def test_iter(self):
        """测试__iter__方法"""
        dataset = MutationDataset()
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加突变集
        mutation_sets_added = []
        for i in range(3):
            mutation = AminoAcidMutation("A", i, "V")
            mutation_set = AminoAcidMutationSet([mutation], name=f"set_{i}")
            dataset.add_mutation_set(mutation_set, "ref1")
            mutation_sets_added.append(mutation_set)

        # 测试迭代
        for i, (mutation_set, ref_id) in enumerate(dataset):
            assert mutation_set.name == f"set_{i}"
            assert ref_id == "ref1"

    def test_str(self):
        """测试__str__方法"""
        dataset = MutationDataset(name="test_dataset")
        protein_seq = ProteinSequence("ACDEFG", name="test_protein")
        dataset.add_reference_sequence("ref1", protein_seq)

        # 添加突变
        mutations = [AminoAcidMutation("A", 0, "V"), AminoAcidMutation("C", 1, "G")]
        mutation_set = AminoAcidMutationSet(mutations)
        dataset.add_mutation_set(mutation_set, "ref1")

        str_repr = str(dataset)
        assert "MutationDataset(test_dataset)" in str_repr
        assert "1 reference sequences" in str_repr
        assert "1 mutation sets" in str_repr
        assert "2 mutations" in str_repr


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dataset_operations(self):
        """测试空数据集的操作"""
        dataset = MutationDataset()

        # 测试空数据集的统计
        stats = dataset.get_statistics()
        assert stats["total_mutation_sets"] == 0
        assert stats["total_mutations"] == 0

        # 测试空数据集的DataFrame转换
        df = dataset.to_dataframe()
        assert df.empty

        # 测试空数据集的验证
        results = dataset.validate_against_references()
        assert len(results["valid_mutation_sets"]) == 0

    def test_mutation_set_index_out_of_range(self):
        """测试越界的突变集索引"""
        dataset = MutationDataset()

        with pytest.raises(ValueError, match="Mutation set index 0 out of range"):
            dataset.get_mutation_set_reference(0)

        with pytest.raises(ValueError, match="Mutation set index 0 out of range"):
            dataset.get_mutation_set_label(0)

        with pytest.raises(ValueError, match="Mutation set index 0 out of range"):
            dataset.set_mutation_set_label(0, 1.0)

    def test_from_dataframe_validation(self):
        """测试从DataFrame创建时的验证"""
        # 空DataFrame
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            MutationDataset.from_dataframe(pd.DataFrame(), {})

        # 缺少必需列
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            MutationDataset.from_dataframe(df, {})

    def test_mixed_sequence_types(self):
        """测试混合序列类型的数据集"""
        dataset = MutationDataset()

        # 添加不同类型的序列
        protein_seq = ProteinSequence("ACDEFG", name="protein")
        dna_seq = DNASequence("ATGGCCTAT", name="dna")
        rna_seq = RNASequence("AUGGCCUAU", name="rna")

        dataset.add_reference_sequence("protein_ref", protein_seq)
        dataset.add_reference_sequence("dna_ref", dna_seq)
        dataset.add_reference_sequence("rna_ref", rna_seq)

        # 添加对应的突变
        aa_mutation = AminoAcidMutation("A", 0, "V")
        aa_set = AminoAcidMutationSet([aa_mutation])
        dataset.add_mutation_set(aa_set, "protein_ref")

        codon_mutation = CodonMutation("ATG", 0, "CTG")
        codon_set = CodonMutationSet([codon_mutation])
        dataset.add_mutation_set(codon_set, "dna_ref")

        # 验证
        assert len(dataset) == 2
        assert len(dataset.reference_sequences) == 3

        # 测试统计
        stats = dataset.get_statistics()
        assert stats["num_reference_sequences"] == 3
