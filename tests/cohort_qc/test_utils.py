from inmoose.cohort_qc.utils import round_scientific_notation, truncate_name


def test_round_scientific_notation(self):
    """Test rounding of small positive numbers."""
    result = round_scientific_notation(0.0000123456)
    self.assertEqual(result, "1.23e-05")


def test_truncate_name(self):
    """Test case where the name is longer than max_length."""
    result = truncate_name("ThisNameIsTooLong", 11)
    self.assertEqual(result, "ThisName...")
