import unittest

from inmoose.data.pasilla import pasilla


class Test(unittest.TestCase):
    def test_pasilla(self):
        """
        check that the pasilla dataset can be properly loaded
        """
        adata = pasilla()

        self.assertEqual(adata.shape, (7, 14599))
        self.assertEqual(len(adata.obs.columns), 5)
