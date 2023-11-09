# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2023 Maximilien Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


class DGEGLM(object):
    def __init__(self, fit):
        # fit is a 5-tuple
        (coefficients, fitted_values, deviance, iter, failed) = fit
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.deviance = deviance
        self.iter = iter
        self.failed = failed

        self.counts = None
        self.design = None
        self.offset = None
        self.dispersion = None
        self.weights = None
        self.prior_count = None
        self.unshrunk_coefficients = None
        self.method = None
