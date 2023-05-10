import unittest
import numpy as np

from inmoose.utils import Factor, rnbinom
from inmoose.edgepy import mglmLevenberg, mglmOneGroup, mglmOneWay, designAsFactor

class test_mglm(unittest.TestCase):

    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20,4))
        self.y = np.vstack(([0,0,0,0], [0,0,2,2], y))
        self.group = np.array([1,1,2,2])
        self.dispersion = 0.05
        self.lib_size = self.y.sum(axis=0)

    def test_mglmOneGroup(self):
        j = self.group == 1
        y1 = mglmOneGroup(self.y[:, j], dispersion=self.dispersion, offset=np.log(self.lib_size[j]))
        ref1 = np.array([np.NINF,np.NINF,-2.986383,-3.485387,-2.896575,-3.676740,-3.215694,-3.056997,-3.406943,-3.423806,-2.740691,-3.226567,-2.741902,-2.648455,-3.053310,-2.971478,-3.166231,-2.754931,-2.635907,-2.494544,-3.060501,-3.192916])
        self.assertTrue(np.allclose(y1, ref1, atol=1e-6, rtol=0))

        j = self.group == 2
        y2 = mglmOneGroup(self.y[:, j], dispersion=self.dispersion, offset=np.log(self.lib_size[j]))
        ref2 = np.array([np.NINF,-5.420406,-3.273714,-3.506488,-3.452560,-2.762558,-2.748622,-2.780652,-3.252326,-2.851513,-3.218678,-3.277784,-3.211742,-3.407176,-2.912961,-2.475169,-3.472935,-3.057142,-2.737961,-2.636946,-2.794320,-3.094864])
        self.assertTrue(np.allclose(y2, ref2, atol=1e-6, rtol=0))

    # TODO also test with varying tolerance
    def test_mglmLevenberg(self):
        design = np.ones((self.y.shape[1], 1))
        coef_ref = np.array([
                           np.nan,
                                0,
            2.9704144655697013455,
            2.6026896854443828389,
            2.9704144655697013455,
            2.9831534913471302595,
            3.1463051320333654814,
            3.1986731175506815106,
            2.7725887222397807008,
            3.0081547935525478898,
             3.135494215929149675,
            2.8478121434773688847,
             3.135494215929149675,
             3.135494215929149675,
            3.1135153092103742267,
            3.4094961844768505443,
             2.788092908775746448,
            3.2188758248682005636,
            3.4094961844768505443,
             3.533686564708234279,
            3.1780538303479461959,
            2.9575110607337933288]).reshape(self.y.shape[0], 1)

        fit_ref = np.array([
                [ 0.00, 0.00, 0.00, 0.00],
                [ 1.00, 1.00, 1.00, 1.00],
                [19.50,19.50,19.50,19.50],
                [13.50,13.50,13.50,13.50],
                [19.50,19.50,19.50,19.50],
                [19.75,19.75,19.75,19.75],
                [23.25,23.25,23.25,23.25],
                [24.50,24.50,24.50,24.50],
                [16.00,16.00,16.00,16.00],
                [20.25,20.25,20.25,20.25],
                [23.00,23.00,23.00,23.00],
                [17.25,17.25,17.25,17.25],
                [23.00,23.00,23.00,23.00],
                [23.00,23.00,23.00,23.00],
                [22.50,22.50,22.50,22.50],
                [30.25,30.25,30.25,30.25],
                [16.25,16.25,16.25,16.25],
                [25.00,25.00,25.00,25.00],
                [30.25,30.25,30.25,30.25],
                [34.25,34.25,34.25,34.25],
                [24.00,24.00,24.00,24.00],
                [19.25,19.25,19.25,19.25]
                ])

        dev_ref = np.array([0.0000000,5.5451767,6.4342458,3.6714251,40.1310924,19.0526072,7.5713327,22.3840805,0.8552751,21.3975973,9.7128441,2.8266627,21.4133481,11.9659419,3.8835243,8.5881018,1.6366558,49.5556034,17.5782768,4.3513363,4.2882649,1.0473488])
        (coef, fit, dev, it, fail) = mglmLevenberg(self.y, design)
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-15, rtol=0, equal_nan=True))
        self.assertTrue(np.allclose(fit, fit_ref, atol=1e-6, rtol=0))
        self.assertTrue(np.allclose(dev, dev_ref, atol=1e-6, rtol=0))
        self.assertEqual(it[0], 0)
        self.assertEqual(np.unique(it[1:]), 1)
        self.assertFalse(np.any(fail))

        (coef, fit, dev, it, fail) = mglmLevenberg(self.y, design, coef_start=np.ones((self.y.shape[0], design.shape[1])))
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-6, rtol=0, equal_nan=True))

        with self.assertRaisesRegex(ValueError, expected_regex="no data"):
            mglmLevenberg(np.ones(shape=(0,1)), None)
        with self.assertRaisesRegex(ValueError, expected_regex="invalid start_method foo"):
            mglmLevenberg(self.y, design, start_method="foo")

class test_mglmOneWay(unittest.TestCase):

    def setUp(self):
        y = np.array(rnbinom(80, size=5, mu=20, seed=42)).reshape((20,4))
        self.y = np.vstack(([0,0,0,0], [0,0,2,2], y))
        self.group = np.array([1,1,2,2])
        self.dispersion = 0.05
        self.lib_size = self.y.sum(axis=0)

    def test_designAsFactor(self):
        f1 = designAsFactor(np.full((3,4), [1,2,3,4]))
        f2 = designAsFactor(np.full((3,4), [1,2,3,4]).T)
        self.assertTrue(np.array_equal(f1.__array__(), Factor([1,1,1]).__array__()))
        self.assertTrue(np.array_equal(f2.__array__(), Factor([1,2,3,4]).__array__()))

    def test_mglmOneWay(self):
        coef_ref = np.array([-1.000000e+08,0.000000e+00,2.970414e+00,2.602690e+00,2.970414e+00,2.983153e+00,3.146305e+00,3.198673e+00,2.772589e+00,3.008155e+00,3.135494e+00,2.847812e+00,3.135494e+00,3.135494e+00,3.113515e+00,3.409496e+00,2.788093e+00,3.218876e+00,3.409496e+00,3.533687e+00,3.178054e+00,2.957511e+00]).reshape(self.y.shape[0],1)
        (coef, fit) = mglmOneWay(self.y)
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-6, rtol=0))

        design = np.array([[1,0],[1,0],[0,1],[0,1]])
        coef_ref = np.array([
            [-1.000000e+08,-1.000000e+08],
            [-1.000000e+08, 6.931472e-01],
            [ 3.091042e+00, 2.833213e+00],
            [ 2.602690e+00, 2.602690e+00],
            [ 3.198673e+00, 2.674149e+00],
            [ 2.397895e+00, 3.349904e+00],
            [ 2.862201e+00, 3.367296e+00],
            [ 3.044522e+00, 3.332205e+00],
            [ 2.674149e+00, 2.862201e+00],
            [ 2.674149e+00, 3.258097e+00],
            [ 3.332205e+00, 2.890372e+00],
            [ 2.862201e+00, 2.833213e+00],
            [ 3.332205e+00, 2.890372e+00],
            [ 3.433987e+00, 2.708050e+00],
            [ 3.020425e+00, 3.198673e+00],
            [ 3.113515e+00, 3.637586e+00],
            [ 2.917771e+00, 2.639057e+00],
            [ 3.349904e+00, 3.068053e+00],
            [ 3.433987e+00, 3.384390e+00],
            [ 3.583519e+00, 3.481240e+00],
            [ 3.020425e+00, 3.314186e+00],
            [ 2.890372e+00, 3.020425e+00]])
        (coef, fit) = mglmOneWay(self.y, design=design)
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-6, rtol=0))

        design = np.array([[2,0],[2,0],[0,2],[0,2]])
        coef_ref = np.array([
            [-5.000000e+07,-5.000000e+07],
            [-5.000000e+07, 3.465736e-01],
            [ 1.545521e+00, 1.416607e+00],
            [ 1.301345e+00, 1.301345e+00],
            [ 1.599337e+00, 1.337074e+00],
            [ 1.198948e+00, 1.674952e+00],
            [ 1.431100e+00, 1.683648e+00],
            [ 1.522261e+00, 1.666102e+00],
            [ 1.337074e+00, 1.431100e+00],
            [ 1.337074e+00, 1.629048e+00],
            [ 1.666102e+00, 1.445186e+00],
            [ 1.431100e+00, 1.416607e+00],
            [ 1.666102e+00, 1.445186e+00],
            [ 1.716994e+00, 1.354025e+00],
            [ 1.510212e+00, 1.599337e+00],
            [ 1.556758e+00, 1.818793e+00],
            [ 1.458885e+00, 1.319529e+00],
            [ 1.674952e+00, 1.534026e+00],
            [ 1.716994e+00, 1.692195e+00],
            [ 1.791759e+00, 1.740620e+00],
            [ 1.510212e+00, 1.657093e+00],
            [ 1.445186e+00, 1.510212e+00]])
        (coef, fit) = mglmOneWay(self.y, design=design)
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-6, rtol=0))
