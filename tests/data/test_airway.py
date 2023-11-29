import unittest

from inmoose.data.airway import airway


class Test(unittest.TestCase):
    def test_airway(self):
        """
        check that the airway dataset is properly loaded
        """
        adata = airway()
        self.assertEqual(adata.shape, (8, 64102))
        self.assertEqual(len(adata.obs.columns), 9)
