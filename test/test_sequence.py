import pytest
import warnings
from unittest.mock import Mock, patch

from tidymut.core.sequence import (
    DNASequence,
    RNASequence,
    ProteinSequence,
    translate,
)
from tidymut.core.mutation import (
    AminoAcidMutation,
    CodonMutation,
    AminoAcidMutationSet,
    CodonMutationSet,
)
from tidymut.core.alphabet import DNAAlphabet, RNAAlphabet, ProteinAlphabet
from tidymut.core.codon import CodonTable


class TestSequenceBase:
    """测试基础Sequence类"""

    def test_sequence_creation(self):
        """测试序列创建"""
        # 创建一个简单的DNA序列用于测试
        alphabet = DNAAlphabet()
        seq = DNASequence("ATCG", alphabet=alphabet, name="test_seq")

        assert len(seq) == 4
        assert str(seq) == "ATCG"
        assert seq.name == "test_seq"
        assert seq.metadata == {}

    def test_sequence_creation_with_metadata(self):
        """测试带元数据的序列创建"""
        metadata = {"source": "test", "version": 1}
        seq = DNASequence("ATCG", name="test", metadata=metadata)

        assert seq.metadata == metadata

    def test_sequence_equality(self):
        """测试序列相等性"""
        seq1 = DNASequence("ATCG")
        seq2 = DNASequence("ATCG")
        seq3 = DNASequence("TTCG")
        seq4 = RNASequence("AUCG")

        assert seq1 == seq2
        assert seq1 != seq3
        assert seq1 != seq4  # 不同类型的alphabet
        assert seq1 == "ATCG"  # 与字符串比较

    def test_get_subsequence_0_indexed(self):
        """测试0-indexed的子序列提取"""
        seq = DNASequence("ATCGATCG", name="test")

        # 测试正常情况
        subseq = seq.get_subsequence(2, 5)
        assert str(subseq) == "CGA"
        assert subseq.name == "test_2_5"

        # 测试没有end参数
        subseq2 = seq.get_subsequence(3)  # 从位置3到末尾
        assert str(subseq2) == "GATCG"
        assert subseq2.name == f"test_3_{len(seq)}"

    def test_get_subsequence_invalid_start(self):
        """测试无效的起始位置"""
        seq = DNASequence("ATCG")

        with pytest.raises(
            IndexError, match="Start position must be greater than or equal to 0"
        ):
            seq.get_subsequence(-1)

    def test_slice(self):
        """test slice"""
        seq = DNASequence("ATCGATCG")
        assert str(seq[:5]) == "ATCGA"  # 0-indexed slice
        assert str(seq[5:]) == "TCG"  # 0-indexed slice
        assert str(seq[::-1]) == "ATCGATCG"[::-1]  # 0-indexed slice, reversed


class TestDNASequence:
    """测试DNA序列类"""

    def test_dna_sequence_creation(self):
        """测试DNA序列创建"""
        seq = DNASequence("ATCGATCG")
        assert str(seq) == "ATCGATCG"
        assert isinstance(seq.alphabet, DNAAlphabet)

    def test_reverse_complement(self):
        """测试DNA反向互补"""
        seq = DNASequence("ATCG", name="test")
        rc = seq.reverse_complement()

        assert str(rc) == "CGAT"  # ATCG -> CGAT
        assert rc.name == "test_rc"
        assert isinstance(rc, DNASequence)

    def test_reverse_complement_complex(self):
        """测试复杂DNA序列的反向互补"""
        seq = DNASequence("ATCGATCGATCG")
        rc = seq.reverse_complement()

        assert str(rc) == "CGATCGATCGAT"

    def test_transcribe(self):
        """测试DNA转录为RNA"""
        seq = DNASequence("ATCGATCG", name="test")
        rna = seq.transcribe()

        assert str(rna) == "AUCGAUCG"  # T -> U
        assert rna.name == "test_transcribed"
        assert isinstance(rna, RNASequence)

    @patch("tidymut.core.sequence.translate")
    def test_dna_translate(self, mock_translate):
        """测试DNA翻译"""
        mock_translate.return_value = "MET"

        seq = DNASequence("ATGATG", name="test")
        protein = seq.translate()

        mock_translate.assert_called_once_with(
            sequence="ATGATG",
            seq_type="DNA",
            codon_table=None,
            start_at_first_met=False,
            stop_at_stop_codon=False,
            require_mod3=True,
            start=None,
            end=None,
        )
        assert str(protein) == "MET"
        assert protein.name == "test_translation"


class TestRNASequence:
    """测试RNA序列类"""

    def test_rna_sequence_creation(self):
        """测试RNA序列创建"""
        seq = RNASequence("AUCGAUCG")
        assert str(seq) == "AUCGAUCG"
        assert isinstance(seq.alphabet, RNAAlphabet)

    def test_reverse_complement(self):
        """测试RNA反向互补"""
        seq = RNASequence("AUCG", name="test")
        rc = seq.reverse_complement()

        assert str(rc) == "CGAU"  # AUCG -> CGAU
        assert rc.name == "test_rc"
        assert isinstance(rc, RNASequence)

    def test_back_transcribe(self):
        """测试RNA反转录为DNA"""
        seq = RNASequence("AUCGAUCG", name="test")
        dna = seq.back_transcribe()

        assert str(dna) == "ATCGATCG"  # U -> T
        assert dna.name == "test_back_transcribe"
        assert isinstance(dna, DNASequence)

    @patch("tidymut.core.sequence.translate")
    def test_rna_translate(self, mock_translate):
        """测试RNA翻译"""
        mock_translate.return_value = "MET"

        seq = RNASequence("AUGUGAUG", name="test")
        protein = seq.translate(start_at_first_met=True)

        mock_translate.assert_called_once_with(
            sequence="AUGUGAUG",
            seq_type="RNA",
            codon_table=None,
            start_at_first_met=True,
            stop_at_stop_codon=False,
            require_mod3=True,
            start=None,
            end=None,
        )


class TestProteinSequence:
    """测试蛋白质序列类"""

    def test_protein_sequence_creation(self):
        """测试蛋白质序列创建"""
        seq = ProteinSequence("METALA")
        assert str(seq) == "METALA"
        assert isinstance(seq.alphabet, ProteinAlphabet)

    def test_get_residue(self):
        """测试获取特定位置的氨基酸"""
        seq = ProteinSequence("METALA")

        assert seq.get_residue(0) == "M"
        assert seq.get_residue(1) == "E"
        assert seq.get_residue(5) == "A"

    def test_get_residue_out_of_range(self):
        """测试越界访问"""
        seq = ProteinSequence("MET")

        with pytest.raises(IndexError):
            seq.get_residue(-1)

        with pytest.raises(IndexError):
            seq.get_residue(3)  # 应该是>=3，因为长度是3

    def test_find_motif(self):
        """测试查找motif"""
        seq = ProteinSequence("METKMETALA")

        # 查找单个motif
        positions = seq.find_motif("MET")
        assert positions == [0, 4]

        # 查找不存在的motif
        positions = seq.find_motif("XYZ")
        assert positions == []

        # 查找单个字符
        positions = seq.find_motif("A")
        assert positions == [7, 9]

    def test_find_motif_case_insensitive(self):
        """测试motif查找是否大小写不敏感"""
        seq = ProteinSequence("METALA")

        positions = seq.find_motif("met")
        assert positions == [0]

        positions = seq.find_motif("Met")
        assert positions == [0]


class TestTranslateFunction:
    """测试翻译函数"""

    @pytest.fixture
    def mock_codon_table(self):
        """模拟密码子表"""
        table = Mock(spec=CodonTable)
        table.is_start_codon.return_value = False
        table.is_stop_codon.return_value = False
        table.translate_codon.side_effect = lambda x: {
            "ATG": "M",
            "GAA": "X",
            "TAG": "*",
        }.get(x, "X")
        return table

    @patch("tidymut.core.codon.CodonTable.get_standard_table")
    def test_translate_basic(self, mock_get_table, mock_codon_table):
        """测试基本翻译功能"""
        mock_get_table.return_value = mock_codon_table

        result = translate("ATGGAA", seq_type="DNA")

        assert result == "MX"  # ATG->M, GAA->E 但mock返回X
        mock_get_table.assert_called_once_with(seq_type="DNA")

    @patch("tidymut.core.codon.CodonTable.get_standard_table")
    def test_translate_with_start_codon(self, mock_get_table, mock_codon_table):
        """测试从起始密码子开始翻译"""
        mock_codon_table.is_start_codon.side_effect = lambda x: x == "ATG"
        mock_get_table.return_value = mock_codon_table

        result = translate("GAAATGGAA", seq_type="DNA", start_at_first_met=True)

        assert result == "MX"  # 从ATG开始

    @patch("tidymut.core.codon.CodonTable.get_standard_table")
    def test_translate_with_stop_codon(self, mock_get_table, mock_codon_table):
        """测试遇到终止密码子停止翻译"""
        mock_codon_table.is_stop_codon.side_effect = lambda x: x == "TAG"
        mock_get_table.return_value = mock_codon_table

        result = translate("ATGTAGGAA", seq_type="DNA", stop_at_stop_codon=True)

        assert result == "M*"  # 在TAG处停止

    def test_translate_not_mod3_strict(self):
        """测试序列长度不是3的倍数时的严格模式"""
        with pytest.raises(ValueError, match="not divisible by 3"):
            translate("ATGGA", require_mod3=True)

    @patch("tidymut.core.sequence.CodonTable.get_standard_table")
    def test_translate_not_mod3_lenient(self, mock_get_table, mock_codon_table):
        """测试序列长度不是3的倍数时的宽松模式"""
        mock_get_table.return_value = mock_codon_table

        with warnings.catch_warnings(record=True) as w:
            result = translate("ATGGA", require_mod3=False)

            assert len(w) == 1
            assert "not divisible by 3" in str(w[0].message)

    @patch("tidymut.core.sequence.CodonTable.get_standard_table")
    def test_translate_custom_positions(self, mock_get_table, mock_codon_table):
        """测试自定义起始和结束位置"""
        mock_get_table.return_value = mock_codon_table

        result = translate("AAATGGAATAG", start=3, end=9)

        # 应该翻译ATG GAA (位置3-8)
        assert len(result) == 2

    @patch("tidymut.core.sequence.CodonTable.get_standard_table")
    def test_translate_no_start_codon_found(self, mock_get_table, mock_codon_table):
        """测试找不到起始密码子的情况"""
        mock_codon_table.is_start_codon.return_value = False
        mock_get_table.return_value = mock_codon_table

        result = translate("GAAGAAGAA", start_at_first_met=True)

        assert result == ""


class TestEdgeCases:
    """测试边界情况和错误处理"""

    def test_empty_sequence(self):
        """测试空序列"""
        # 这可能会根据你的验证逻辑抛出异常
        try:
            seq = DNASequence("")
            assert len(seq) == 0
        except ValueError:
            # 如果你的代码不允许空序列，这是预期的
            pass

    def test_invalid_characters(self):
        """测试无效字符（取决于alphabet的验证）"""
        # 这取决于你的alphabet验证实现
        try:
            seq = DNASequence("ATCGXYZ")
            # 如果通过了，说明alphabet允许这些字符
        except ValueError:
            # 如果抛出异常，说明alphabet有适当的验证
            pass

    def test_none_name_handling(self):
        """测试None名称的处理"""
        seq = DNASequence("ATCG", name=None)
        subseq = seq.get_subsequence(1, 2)

        assert subseq.name is None

    def test_metadata_preservation(self):
        """测试元数据在操作中的保持"""
        metadata = {"test": "value"}
        seq = DNASequence("ATCG", metadata=metadata)

        rc = seq.reverse_complement()
        rna = seq.transcribe()

        assert rc.metadata == metadata
        assert rna.metadata == metadata


class TestPerformance:
    """性能相关测试"""

    def test_large_sequence_operations(self):
        """测试大序列操作"""
        # 创建一个相对较大的序列
        large_seq = DNASequence("ATCG" * 1000)  # 4000个字符

        # 这些操作应该能够快速完成
        assert len(large_seq) == 4000

        rc = large_seq.reverse_complement()
        assert len(rc) == 4000

        subseq = large_seq.get_subsequence(1, 100)
        assert len(subseq) == 99

    def test_motif_finding_performance(self):
        """测试motif查找性能"""
        seq = ProteinSequence("MET" * 100)  # 重复的MET

        positions = seq.find_motif("MET")
        assert len(positions) == 100


class TestApplyMutation:
    """测试apply_mutation方法"""

    def test_apply_amino_acid_mutation(self):
        """测试应用氨基酸突变"""
        seq = ProteinSequence("METALA", name="test")
        mutation = AminoAcidMutation("M", 0, "V")  # M0V

        mutated_seq = seq.apply_mutation(mutation)

        assert str(mutated_seq) == "VETALA"
        assert mutated_seq.name == "test"  # 名称保持不变
        assert "mutations_applied" in mutated_seq.metadata
        assert len(mutated_seq.metadata["mutations_applied"]) == 1

        mutation_record = mutated_seq.metadata["mutations_applied"][0]
        assert mutation_record["type"] == "AminoAcidMutation"
        assert mutation_record["position"] == 0
        assert mutation_record["wild_amino_acid"] == "M"
        assert mutation_record["mutant_amino_acid"] == "V"

    def test_apply_codon_mutation_dna(self):
        """测试应用DNA密码子突变"""
        seq = DNASequence("ATGAAATAG", name="test")
        mutation = CodonMutation("ATG", 0, "TTG")  # ATG0TTG

        mutated_seq = seq.apply_mutation(mutation)

        assert str(mutated_seq) == "TTGAAATAG"
        assert mutated_seq.name == "test"
        assert "mutations_applied" in mutated_seq.metadata

        mutation_record = mutated_seq.metadata["mutations_applied"][0]
        assert mutation_record["type"] == "CodonMutation"
        assert mutation_record["wild_codon"] == "ATG"
        assert mutation_record["mutant_codon"] == "TTG"

    def test_apply_codon_mutation_rna(self):
        """测试应用RNA密码子突变"""
        seq = RNASequence("AUGUGAAAG", name="test")
        mutation = CodonMutation("AUG", 0, "UUG")  # AUG0UUG

        mutated_seq = seq.apply_mutation(mutation)

        assert str(mutated_seq) == "UUGUGAAAG"

    def test_apply_amino_acid_mutation_set(self):
        """测试应用氨基酸突变集合"""
        seq = ProteinSequence("METALA")
        mutations = [
            AminoAcidMutation("M", 0, "V"),  # M0V
            AminoAcidMutation("A", 3, "G"),  # A3G
        ]
        mutation_set = AminoAcidMutationSet(mutations)

        mutated_seq = seq.apply_mutation(mutation_set)

        assert str(mutated_seq) == "VETGLA"
        assert len(mutated_seq.metadata["mutations_applied"]) == 2

    def test_apply_codon_mutation_set(self):
        """测试应用密码子突变集合"""
        seq = DNASequence("ATGAAACCC")
        mutations = [
            CodonMutation("ATG", 0, "TTG"),  # ATG0TTG
            CodonMutation("CCC", 6, "GGG"),  # CCC6GGG
        ]
        mutation_set = CodonMutationSet(mutations)

        mutated_seq = seq.apply_mutation(mutation_set)

        assert str(mutated_seq) == "TTGAAAGGG"

    def test_mutation_position_validation_amino_acid(self):
        """测试氨基酸突变位置验证"""
        seq = ProteinSequence("MET")

        # 测试位置越界
        mutation = AminoAcidMutation("A", 5, "V")  # 位置5超出范围
        with pytest.raises(ValueError, match="out of bounds"):
            seq.apply_mutation(mutation)

    def test_mutation_position_validation_codon(self):
        """测试密码子突变位置验证"""
        seq = DNASequence("ATGAAA")  # 长度6

        # 测试密码子超出边界
        mutation = CodonMutation("AAA", 4, "TTT")  # 位置4+3=7 > 6
        with pytest.raises(ValueError, match="extends beyond sequence length"):
            seq.apply_mutation(mutation)

    def test_original_sequence_validation_amino_acid(self):
        """测试原始氨基酸序列验证"""
        seq = ProteinSequence("METALA")

        # 原始氨基酸不匹配
        mutation = AminoAcidMutation("K", 0, "V")  # 位置0是M，不是K
        with pytest.raises(ValueError, match="Expected amino acid 'K'"):
            seq.apply_mutation(mutation)

    def test_original_sequence_validation_codon(self):
        """测试原始密码子序列验证"""
        seq = DNASequence("ATGAAATAG")

        # 原始密码子不匹配
        mutation = CodonMutation("TTG", 0, "GGG")  # 位置0是ATG，不是TTG
        with pytest.raises(ValueError, match="Expected codon 'TTG'"):
            seq.apply_mutation(mutation)

    def test_multiple_mutations_order(self):
        """测试多个突变的应用顺序（逆序避免位置偏移）"""
        seq = ProteinSequence("METALA")
        mutations = [
            AminoAcidMutation("M", 0, "V"),  # 位置0
            AminoAcidMutation("T", 2, "S"),  # 位置2
            AminoAcidMutation("A", 3, "G"),  # 位置3
        ]
        mutation_set = AminoAcidMutationSet(mutations)

        mutated_seq = seq.apply_mutation(mutation_set)

        # 应该是从位置大到小应用: A3G -> T2S -> M0V
        assert str(mutated_seq) == "VESGLA"

    def test_metadata_preservation(self):
        """测试原有元数据的保留"""
        original_metadata = {"source": "test", "version": 1}
        seq = ProteinSequence("MET", metadata=original_metadata)
        mutation = AminoAcidMutation("M", 0, "V")

        mutated_seq = seq.apply_mutation(mutation)

        # 原有元数据应该保留
        assert mutated_seq.metadata["source"] == "test"
        assert mutated_seq.metadata["version"] == 1
        # 新的突变记录也应该存在
        assert "mutations_applied" in mutated_seq.metadata

    def test_unsupported_mutation_type(self):
        """测试不支持的突变类型"""
        seq = ProteinSequence("MET")

        # 模拟一个不支持的突变类型
        unsupported_mutation = Mock()
        unsupported_mutation.__class__.__name__ = "UnsupportedMutation"

        with pytest.raises(TypeError, match="Unsupported mutation type"):
            seq.apply_mutation(unsupported_mutation)

    def test_immutability(self):
        """测试原序列不变性"""
        seq = ProteinSequence("MET")
        mutation = AminoAcidMutation("M", 0, "V")

        mutated_seq = seq.apply_mutation(mutation)

        # 原序列应该保持不变
        assert str(seq) == "MET"
        assert str(mutated_seq) == "VET"
        assert seq is not mutated_seq

    def test_chained_mutations(self):
        """测试链式突变应用"""
        seq = ProteinSequence("METALA")
        mutation1 = AminoAcidMutation("M", 0, "V")
        mutation2 = AminoAcidMutation("E", 1, "K")

        # 链式应用
        mutated_seq = seq.apply_mutation(mutation1).apply_mutation(mutation2)

        assert str(mutated_seq) == "VKTALA"
        # 应该有两次突变记录
        assert len(mutated_seq.metadata["mutations_applied"]) == 2

    def test_unmatched_mutation_subtype_and_sequence_type(self):
        """测试类型不匹配的报错"""
        seq = RNASequence("AUGUGAAAG", name="test")
        unsupported_mutation = CodonMutation("ATG", 0, "AAG")  # subtype: codon_dna

        with pytest.raises(TypeError, match="Unmatching mutation subtype"):
            seq.apply_mutation(unsupported_mutation)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
