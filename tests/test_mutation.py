import pytest

from tidymut.core.mutation import (
    BaseMutation,
    AminoAcidMutation,
    CodonMutation,
    MutationSet,
    AminoAcidMutationSet,
    CodonMutationSet,
)
from tidymut.core.alphabet import ProteinAlphabet
from tidymut.core.codon import CodonTable


class TestBaseMutation:
    """测试基础Mutation类"""

    def test_base_mutation_is_abstract(self):
        """测试BaseMutation是抽象类"""
        with pytest.raises(TypeError):
            BaseMutation(position=1)  # type: ignore

    def test_invalid_position(self):
        """测试无效位置"""
        with pytest.raises(ValueError, match="Position must be non-negative"):
            AminoAcidMutation("A", -1, "V")


class TestAminoAcidMutation:
    """测试氨基酸突变类"""

    def test_amino_acid_mutation_creation(self):
        """测试氨基酸突变创建"""
        mutation = AminoAcidMutation("A", 123, "V")

        assert mutation.wild_amino_acid == "A"
        assert mutation.position == 123
        assert mutation.mutant_amino_acid == "V"
        assert mutation.type == "amino_acid"
        assert str(mutation) == "A123V"

    def test_amino_acid_mutation_with_metadata(self):
        """测试带元数据的氨基酸突变"""
        metadata = {"source": "test", "confidence": 0.95}
        mutation = AminoAcidMutation("A", 123, "V", metadata=metadata)

        assert mutation.metadata == metadata

    def test_amino_acid_mutation_with_custom_alphabet(self):
        """测试使用自定义字母表的氨基酸突变"""
        alphabet = ProteinAlphabet(include_stop=False)
        mutation = AminoAcidMutation("A", 123, "V", alphabet=alphabet)

        assert mutation.alphabet == alphabet

    def test_amino_acid_mutation_case_insensitive(self):
        """测试氨基酸突变大小写不敏感"""
        mutation = AminoAcidMutation("a", 123, "v")

        assert mutation.wild_amino_acid == "A"
        assert mutation.mutant_amino_acid == "V"

    def test_amino_acid_mutation_validation(self):
        """测试氨基酸突变验证"""
        # 测试无效氨基酸
        with pytest.raises(ValueError, match="Invalid amino acid mutation"):
            AminoAcidMutation("X", 123, "V")

        # 测试无效位置（-1位置）
        with pytest.raises(ValueError, match="Position must be non-negative"):
            AminoAcidMutation("A", -1, "V")

    def test_amino_acid_mutation_types(self):
        """测试氨基酸突变类型分类"""
        # 同义突变
        synonymous = AminoAcidMutation("A", 123, "A")
        assert synonymous.is_synonymous()
        assert not synonymous.is_missense()
        assert not synonymous.is_nonsense()
        assert synonymous.get_mutation_category() == "synonymous"
        assert synonymous.effect_type == "synonymous"

        # 错义突变
        missense = AminoAcidMutation("A", 123, "V")
        assert not missense.is_synonymous()
        assert missense.is_missense()
        assert not missense.is_nonsense()
        assert missense.get_mutation_category() == "missense"

        # 无义突变（终止密码子）
        nonsense = AminoAcidMutation("A", 123, "*")
        assert not nonsense.is_synonymous()
        assert not nonsense.is_missense()
        assert nonsense.is_nonsense()
        assert nonsense.get_mutation_category() == "nonsense"

    def test_amino_acid_mutation_equality(self):
        """测试氨基酸突变相等性"""
        mutation1 = AminoAcidMutation("A", 123, "V")
        mutation2 = AminoAcidMutation("A", 123, "V")
        mutation3 = AminoAcidMutation("A", 124, "V")
        mutation4 = AminoAcidMutation("A", 123, "L")

        assert mutation1 == mutation2
        assert mutation1 != mutation3
        assert mutation1 != mutation4

    def test_amino_acid_mutation_hash(self):
        """测试氨基酸突变可哈希"""
        mutation1 = AminoAcidMutation("A", 123, "V")
        mutation2 = AminoAcidMutation("A", 123, "V")
        mutation3 = AminoAcidMutation("A", 124, "V")

        assert hash(mutation1) == hash(mutation2)
        assert hash(mutation1) != hash(mutation3)

        # 测试可以作为set元素
        mutation_set = {mutation1, mutation2, mutation3}
        assert len(mutation_set) == 2

    def test_amino_acid_mutation_from_string_one_letter(self):
        """测试从单字母字符串解析氨基酸突变"""
        mutation = AminoAcidMutation.from_string("A123V")

        assert mutation.wild_amino_acid == "A"
        assert mutation.position == 123
        assert mutation.mutant_amino_acid == "V"

    def test_amino_acid_mutation_from_string_three_letter(self):
        """测试从三字母字符串解析氨基酸突变"""
        mutation = AminoAcidMutation.from_string("Ala123Val")

        assert mutation.wild_amino_acid == "A"
        assert mutation.position == 123
        assert mutation.mutant_amino_acid == "V"

    def test_amino_acid_mutation_from_string_with_stop(self):
        """测试包含终止密码子的氨基酸突变解析"""
        mutation = AminoAcidMutation.from_string("A123*")

        assert mutation.wild_amino_acid == "A"
        assert mutation.position == 123
        assert mutation.mutant_amino_acid == "*"
        assert mutation.is_nonsense()

    def test_amino_acid_mutation_from_string_invalid_format(self):
        """测试无效格式的氨基酸突变解析"""
        invalid_formats = [
            "A123",  # 缺少突变氨基酸
            "123V",  # 缺少野生型氨基酸
            "ABC123V",  # 无效格式
            "A-123V",  # 无效字符
            "",  # 空字符串
            "   ",  # 只有空格
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValueError, match="Invalid mutation format"):
                AminoAcidMutation.from_string(invalid_format)

    def test_amino_acid_mutation_from_string_invalid_amino_acid(self):
        """测试无效氨基酸代码的解析"""
        with pytest.raises(ValueError, match="Unknown three-letter amino acid code"):
            AminoAcidMutation.from_string("Xxx123Val")


class TestCodonMutation:
    """测试密码子突变类"""

    def test_codon_mutation_creation_dna(self):
        """测试DNA密码子突变创建"""
        mutation = CodonMutation("ATG", 1, "TAA")

        assert mutation.wild_codon == "ATG"
        assert mutation.position == 1
        assert mutation.mutant_codon == "TAA"
        assert mutation.seq_type == "DNA"
        assert mutation.type == "codon_dna"
        assert str(mutation) == "ATG1TAA"

    def test_codon_mutation_creation_rna(self):
        """测试RNA密码子突变创建"""
        mutation = CodonMutation("AUG", 1, "UAA")

        assert mutation.wild_codon == "AUG"
        assert mutation.position == 1
        assert mutation.mutant_codon == "UAA"
        assert mutation.seq_type == "RNA"
        assert mutation.type == "codon_rna"

    def test_codon_mutation_creation_both(self):
        """测试无T/U的密码子突变"""
        mutation = CodonMutation("ACG", 1, "GCA")

        assert mutation.seq_type == "Both"
        assert mutation.type == "codon_both"

    def test_codon_mutation_mixed_tu_error(self):
        """测试混合T和U的错误"""
        with pytest.raises(ValueError, match="Codons cannot contain both T and U"):
            CodonMutation("ATG", 1, "UAA")

    def test_codon_mutation_case_insensitive(self):
        """测试密码子突变大小写不敏感"""
        mutation = CodonMutation("atg", 1, "taa")

        assert mutation.wild_codon == "ATG"
        assert mutation.mutant_codon == "TAA"

    def test_codon_mutation_validation(self):
        """测试密码子突变验证"""
        # 测试无效长度
        with pytest.raises(ValueError, match="Invalid codon mutation"):
            CodonMutation("AT", 1, "TAA")

        with pytest.raises(ValueError, match="Invalid codon mutation"):
            CodonMutation("ATG", 1, "TA")

        # 测试无效字符
        with pytest.raises(ValueError, match="Invalid codon mutation"):
            CodonMutation("ATX", 1, "TAA")

        # 测试无效位置
        with pytest.raises(ValueError, match="Position must be non-negative"):
            CodonMutation("ATG", -1, "TAA")

        with pytest.raises(ValueError, match="Invalid codon mutation"):
            CodonMutation("ATG", 1.4, "TAA")  # type: ignore

    def test_codon_mutation_to_amino_acid(self):
        """测试密码子突变转氨基酸突变"""
        codon_mutation = CodonMutation("ATG", 1, "TAA")  # Met -> Stop
        aa_mutation = codon_mutation.to_amino_acid_mutation()

        assert aa_mutation.wild_amino_acid == "M"
        assert aa_mutation.position == 1
        assert aa_mutation.mutant_amino_acid == "*"
        assert aa_mutation.is_nonsense()

    def test_codon_mutation_to_amino_acid_with_custom_table(self):
        """测试使用自定义密码子表转换"""
        codon_table = CodonTable.get_standard_table("DNA")
        codon_mutation = CodonMutation("ATG", 1, "TTG")  # Met -> Leu
        aa_mutation = codon_mutation.to_amino_acid_mutation(codon_table)

        assert aa_mutation.wild_amino_acid == "M"
        assert aa_mutation.mutant_amino_acid == "L"

    def test_codon_mutation_from_string(self):
        """测试从字符串解析密码子突变"""
        mutation = CodonMutation.from_string("ATG123TAA")

        assert mutation.wild_codon == "ATG"
        assert mutation.position == 123
        assert mutation.mutant_codon == "TAA"

    def test_codon_mutation_from_string_rna(self):
        """测试从RNA字符串解析密码子突变"""
        mutation = CodonMutation.from_string("AUG123UAA")

        assert mutation.wild_codon == "AUG"
        assert mutation.position == 123
        assert mutation.mutant_codon == "UAA"
        assert mutation.seq_type == "RNA"

    def test_codon_mutation_from_string_invalid_format(self):
        """测试无效格式的密码子突变解析"""
        invalid_formats = [
            "ATG123",  # 缺少突变密码子
            "123TAA",  # 缺少野生型密码子
            "AT123TAA",  # 密码子长度不对
            "ATG123TAAA",  # 密码子长度不对
            "",  # 空字符串
            "ATX123TAA",  # 无效字符
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValueError, match="Invalid codon mutation format"):
                CodonMutation.from_string(invalid_format)


class TestMutationSet:
    """测试突变集合类"""

    def test_mutation_set_creation(self):
        """测试突变集合创建"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "P"),
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        assert len(mutation_set) == 2
        assert mutation_set.mutation_type == AminoAcidMutation
        assert mutation_set.mutation_subtype == "amino_acid"

    def test_mutation_set_auto_type_inference(self):
        """测试突变类型自动推断"""
        mutations = [AminoAcidMutation("A", 1, "V")]
        mutation_set = MutationSet(mutations, None)

        assert mutation_set.mutation_type == AminoAcidMutation

    def test_mutation_set_empty_error(self):
        """测试空突变集合错误"""
        with pytest.raises(
            ValueError, match="MutationSet must contain at least one mutation"
        ):
            MutationSet([], AminoAcidMutation)

    def test_mutation_set_type_consistency(self):
        """测试突变类型一致性"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            CodonMutation("ATG", 2, "TAA"),  # 不同类型
        ]

        with pytest.raises(ValueError, match="All mutations must be of type"):
            MutationSet(mutations, AminoAcidMutation)

    def test_mutation_set_subtype_consistency(self):
        """测试突变子类型一致性"""
        mutations = [
            CodonMutation("ATG", 1, "TAA"),  # DNA codon
            CodonMutation("AUG", 2, "UAA"),  # RNA codon
        ]

        with pytest.raises(
            ValueError, match="All mutations must have the same type property"
        ):
            MutationSet(mutations, CodonMutation)

    def test_mutation_set_duplicate_positions(self):
        """测试重复位置错误"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 1, "P"),  # 相同位置
        ]

        with pytest.raises(ValueError, match="Duplicate mutations at positions"):
            MutationSet(mutations, AminoAcidMutation)

    def test_mutation_set_sorting(self):
        """测试突变按位置排序"""
        mutations = [
            AminoAcidMutation("L", 3, "P"),
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("G", 2, "S"),
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        positions = [m.position for m in mutation_set.mutations]
        assert positions == [1, 2, 3]

    def test_mutation_set_iteration(self):
        """测试突变集合迭代"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "P"),
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        mutation_list = list(mutation_set)
        assert len(mutation_list) == 2
        assert mutation_list[0].position == 1
        assert mutation_list[1].position == 2

    def test_mutation_set_add_mutation(self):
        """测试添加突变"""
        mutations = [AminoAcidMutation("A", 1, "V")]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        new_mutation = AminoAcidMutation("L", 2, "P")
        mutation_set.add_mutation(new_mutation)

        assert len(mutation_set) == 2
        assert mutation_set.get_mutation_at(2) == new_mutation

    def test_mutation_set_add_wrong_type(self):
        """测试添加错误类型的突变"""
        mutations = [AminoAcidMutation("A", 1, "V")]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        codon_mutation = CodonMutation("ATG", 2, "TAA")
        with pytest.raises(ValueError, match="Mutation must be of type"):
            mutation_set.add_mutation(codon_mutation)  # type: ignore

    def test_mutation_set_add_duplicate_position(self):
        """测试添加重复位置的突变"""
        mutations = [AminoAcidMutation("A", 1, "V")]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        duplicate_mutation = AminoAcidMutation("A", 1, "L")
        with pytest.raises(ValueError, match="Mutation already exists at position"):
            mutation_set.add_mutation(duplicate_mutation)

    def test_mutation_set_remove_mutation(self):
        """测试移除突变"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "P"),
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        removed = mutation_set.remove_mutation(1)
        assert removed is True
        assert len(mutation_set) == 1
        assert not mutation_set.has_mutation_at(1)

        # 移除不存在的位置
        removed = mutation_set.remove_mutation(999)
        assert removed is False

    def test_mutation_set_get_mutation_at(self):
        """测试获取指定位置的突变"""
        mutation = AminoAcidMutation("A", 1, "V")
        mutations = [mutation]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        assert mutation_set.get_mutation_at(1) == mutation
        assert mutation_set.get_mutation_at(999) is None

    def test_mutation_set_positions(self):
        """测试获取位置信息"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 3, "P"),
            AminoAcidMutation("G", 2, "S"),
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        positions = mutation_set.get_positions()
        assert positions == [1, 2, 3]  # 应该是排序后的

        positions_set = mutation_set.get_positions_set()
        assert positions_set == {1, 2, 3}

    def test_mutation_set_validation(self):
        """测试突变集合验证"""
        mutations = [AminoAcidMutation("A", 1, "V")]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        assert mutation_set.validate_all() is True

    def test_mutation_set_categories(self):
        """测试突变分类统计"""
        mutations = [
            AminoAcidMutation("A", 1, "A"),  # synonymous
            AminoAcidMutation("L", 2, "P"),  # missense
            AminoAcidMutation("G", 3, "*"),  # nonsense
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        categories = mutation_set.get_mutation_categories()
        assert categories["synonymous"] == 1
        assert categories["missense"] == 1
        assert categories["nonsense"] == 1

    def test_mutation_set_filter_by_category(self):
        """测试按分类过滤突变"""
        mutations = [
            AminoAcidMutation("A", 1, "A"),  # synonymous
            AminoAcidMutation("L", 2, "P"),  # missense
            AminoAcidMutation("G", 3, "*"),  # nonsense
        ]
        mutation_set = MutationSet(mutations, AminoAcidMutation)

        missense_mutations = mutation_set.filter_by_category("missense")
        assert len(missense_mutations) == 1
        assert missense_mutations[0].position == 2

    def test_mutation_set_from_string_single(self):
        """测试从字符串创建单个突变集合"""
        mutation_set = MutationSet.from_string("A123V")

        assert len(mutation_set) == 1
        assert mutation_set.mutation_type == AminoAcidMutation
        assert mutation_set.mutations[0].position == 123

    def test_mutation_set_from_string_multiple_comma(self):
        """测试从逗号分隔字符串创建突变集合"""
        mutation_set = MutationSet.from_string("A123V,L456P,G789S")

        assert len(mutation_set) == 3
        assert mutation_set.mutation_type == AminoAcidMutation
        positions = [m.position for m in mutation_set.mutations]
        assert positions == [123, 456, 789]

    def test_mutation_set_from_string_multiple_semicolon(self):
        """测试从分号分隔字符串创建突变集合"""
        mutation_set = MutationSet.from_string("A123V;L456P;G789S")

        assert len(mutation_set) == 3
        positions = [m.position for m in mutation_set.mutations]
        assert positions == [123, 456, 789]

    def test_mutation_set_from_string_with_spaces(self):
        """测试包含空格的字符串"""
        mutation_set = MutationSet.from_string(" A123V , L456P , G789S ")

        assert len(mutation_set) == 3

    def test_mutation_set_from_string_custom_separator(self):
        """测试自定义分隔符"""
        mutation_set = MutationSet.from_string("A123V|L456P|G789S", sep="|")

        assert len(mutation_set) == 3

    def test_mutation_set_from_string_specified_type(self):
        """测试指定突变类型"""
        mutation_set = MutationSet.from_string("ATG123TAA", mutation_type=CodonMutation)

        assert len(mutation_set) == 1
        assert mutation_set.mutation_type == CodonMutation

    def test_mutation_set_from_string_with_name_metadata(self):
        """测试带名称和元数据的字符串创建"""
        metadata = {"source": "test"}
        mutation_set = MutationSet.from_string(
            "A123V", name="test_set", metadata=metadata
        )

        assert mutation_set.name == "test_set"
        assert mutation_set.metadata == metadata

    def test_mutation_set_from_string_empty_error(self):
        """测试空字符串错误"""
        with pytest.raises(ValueError, match="Input string cannot be empty"):
            MutationSet.from_string("")

        with pytest.raises(ValueError, match="Input string cannot be empty"):
            MutationSet.from_string("   ")

    def test_mutation_set_from_string_parse_error(self):
        """测试解析错误"""
        with pytest.raises(ValueError, match="Some mutations could not be parsed"):
            MutationSet.from_string("A123V,INVALID,L456P")

    def test_mutation_set_from_string_no_valid_mutations(self):
        """测试没有有效突变"""
        with pytest.raises(ValueError, match="No valid mutations could be parsed"):
            MutationSet.from_string("INVALID1,INVALID2")

    def test_mutation_set_str_repr(self):
        """测试字符串表示"""
        mutations = [AminoAcidMutation("A", 1, "V")]

        # 无名称
        mutation_set = MutationSet(mutations, AminoAcidMutation)
        assert "MutationSet: A1V" in str(mutation_set)

        # 有名称
        named_set = MutationSet(mutations, AminoAcidMutation, name="test")
        assert "MutationSet(test): A1V" in str(named_set)

        # repr
        repr_str = repr(mutation_set)
        assert "MutationSet(mutations=" in repr_str


class TestAminoAcidMutationSet:
    """测试氨基酸突变集合类"""

    def test_amino_acid_mutation_set_creation(self):
        """测试氨基酸突变集合创建"""
        mutations = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "P"),
        ]
        aa_set = AminoAcidMutationSet(mutations)

        assert len(aa_set) == 2
        assert aa_set.mutation_type == AminoAcidMutation

    def test_amino_acid_mutation_set_effect_filters(self):
        """测试按效应类型过滤"""
        mutations = [
            AminoAcidMutation("A", 1, "A"),  # synonymous
            AminoAcidMutation("L", 2, "P"),  # missense
            AminoAcidMutation("G", 3, "*"),  # nonsense
            AminoAcidMutation("T", 4, "S"),  # missense
        ]
        aa_set = AminoAcidMutationSet(mutations)

        synonymous = aa_set.get_synonymous_mutations()
        assert len(synonymous) == 1
        assert synonymous[0].position == 1

        missense = aa_set.get_missense_mutations()
        assert len(missense) == 2
        assert {m.position for m in missense} == {2, 4}

        nonsense = aa_set.get_nonsense_mutations()
        assert len(nonsense) == 1
        assert nonsense[0].position == 3

    def test_amino_acid_mutation_set_stop_codon_check(self):
        """测试终止密码子检查"""
        mutations_with_stop = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "*"),
        ]
        aa_set_with_stop = AminoAcidMutationSet(mutations_with_stop)
        assert aa_set_with_stop.has_stop_codon_mutations() is True

        mutations_without_stop = [
            AminoAcidMutation("A", 1, "V"),
            AminoAcidMutation("L", 2, "P"),
        ]
        aa_set_without_stop = AminoAcidMutationSet(mutations_without_stop)
        assert aa_set_without_stop.has_stop_codon_mutations() is False

    def test_amino_acid_mutation_set_count_by_effect(self):
        """测试按效应类型计数"""
        mutations = [
            AminoAcidMutation("A", 1, "A"),  # synonymous
            AminoAcidMutation("L", 2, "P"),  # missense
            AminoAcidMutation("G", 3, "*"),  # nonsense
            AminoAcidMutation("T", 4, "S"),  # missense
        ]
        aa_set = AminoAcidMutationSet(mutations)

        counts = aa_set.count_by_effect_type()
        assert counts["synonymous"] == 1
        assert counts["missense"] == 2
        assert counts["nonsense"] == 1


class TestCodonMutationSet:
    """测试密码子突变集合类"""

    def test_codon_mutation_set_creation(self):
        """测试密码子突变集合创建"""
        mutations = [
            CodonMutation("ATG", 1, "TAA"),
            CodonMutation("TTG", 2, "TAG"),
        ]
        codon_set = CodonMutationSet(mutations)

        assert len(codon_set) == 2
        assert codon_set.mutation_type == CodonMutation
        assert codon_set.seq_type == "DNA"

    def test_codon_mutation_set_seq_type(self):
        """测试序列类型属性"""
        # DNA mutations
        dna_mutations = [CodonMutation("ATG", 1, "TAA")]
        dna_set = CodonMutationSet(dna_mutations)
        assert dna_set.seq_type == "DNA"

        # RNA mutations
        rna_mutations = [CodonMutation("AUG", 1, "UAA")]
        rna_set = CodonMutationSet(rna_mutations)
        assert rna_set.seq_type == "RNA"

        # Both mutations
        both_mutations = [CodonMutation("ACG", 1, "GCA")]
        both_set = CodonMutationSet(both_mutations)
        assert both_set.seq_type == "Both"

    def test_codon_mutation_set_to_amino_acid(self):
        """测试转换为氨基酸突变集合"""
        mutations = [
            CodonMutation("ATG", 1, "TAA"),  # Met -> Stop
            CodonMutation("TTG", 2, "CTG"),  # Leu -> Leu
        ]
        codon_set = CodonMutationSet(mutations, name="test_codons")

        aa_set = codon_set.to_amino_acid_mutation_set()

        assert isinstance(aa_set, AminoAcidMutationSet)
        assert len(aa_set) == 2
        assert aa_set.name == "test_codons_aa"

        # 检查转换结果
        aa_mutations = list(aa_set)
        assert aa_mutations[0].wild_amino_acid == "M"
        assert aa_mutations[0].mutant_amino_acid == "*"
        assert aa_mutations[1].wild_amino_acid == "L"
        assert aa_mutations[1].mutant_amino_acid == "L"

    def test_codon_mutation_set_to_amino_acid_custom_table(self):
        """测试使用自定义密码子表转换"""
        mutations = [CodonMutation("ATG", 1, "TTG")]
        codon_set = CodonMutationSet(mutations)

        codon_table = CodonTable.get_standard_table("DNA")
        aa_set = codon_set.to_amino_acid_mutation_set(codon_table)

        aa_mutation = list(aa_set)[0]
        assert aa_mutation.wild_amino_acid == "M"
        assert aa_mutation.mutant_amino_acid == "L"


class TestMutationInference:
    """测试突变类型推断"""

    def test_infer_amino_acid_mutation(self):
        """测试推断氨基酸突变"""
        mutation = MutationSet._infer_and_create_mutation("A123V")
        assert isinstance(mutation, AminoAcidMutation)

    def test_infer_codon_mutation(self):
        """测试推断密码子突变"""
        mutation = MutationSet._infer_and_create_mutation("ATG123TAA")
        assert isinstance(mutation, CodonMutation)

    def test_infer_unknown_mutation(self):
        """测试推断未知突变格式"""
        with pytest.raises(ValueError, match="Could not parse mutation string"):
            MutationSet._infer_and_create_mutation("INVALID_FORMAT")


class TestSeparatorGuessing:
    """测试分隔符猜测"""

    def test_guess_separator_comma(self):
        """测试猜测逗号分隔符"""
        sep = MutationSet._guess_sep("A123V,L456P,G789S")
        assert sep == ","

    def test_guess_separator_semicolon(self):
        """测试猜测分号分隔符"""
        sep = MutationSet._guess_sep("A123V;L456P;G789S")
        assert sep == ";"

    def test_guess_separator_pipe(self):
        """测试猜测管道分隔符"""
        sep = MutationSet._guess_sep("A123V|L456P|G789S")
        assert sep == "|"

    def test_guess_separator_none(self):
        """测试无分隔符情况"""
        sep = MutationSet._guess_sep("A123V")
        assert sep is None

    def test_guess_separator_empty(self):
        """测试空字符串"""
        sep = MutationSet._guess_sep("")
        assert sep is None

    def test_guess_separator_priority(self):
        """测试分隔符优先级"""
        # 当多个分隔符数量相同时，应该选择优先级更高的
        sep = MutationSet._guess_sep("A123V;L456P,G789S")  # 分号和逗号各1个
        assert sep == ";"  # 分号优先级更高
