import unittest

from inmoose.sim.splat import sim_rnaseq

N = 1000
M = 10000
batch = (M // 2) * ["A"] + (M // 2) * ["B"]
group = (M // 4) * ["A"] + (M // 4) * ["B"]
group = group + group


class Test(unittest.TestCase):
    def test_sim_rnaseq(self):
        """
        check that RNA-Seq data simulation works properly
        """
        data = sim_rnaseq(N, M, random_state=42)
        self.assertEqual(data.shape, (N, M))
        data = sim_rnaseq(N, M, batch=batch)
        data = sim_rnaseq(N, M, group=group)

    def test_sim_scrnaseq(self):
        """
        check that scRNA-Seq data simulation works properly
        """
        data = sim_rnaseq(N, M, single_cell=True, random_state=51)
        self.assertEqual(data.shape, (N, M))
