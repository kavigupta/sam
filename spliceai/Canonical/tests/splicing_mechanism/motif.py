import re
import unittest

from parameterized import parameterized

import numpy as np
from modular_splicing.utils.sequence_utils import draw_bases

from working.synthetic_data.splicing_mechanism.motif.psam_motif import PSAMMotif


class MotifTest(unittest.TestCase):
    @parameterized.expand([(x,) for x in range(10)])
    def test_motif(self, ap):
        regex = r"(?=(...(ACGT|TGCA)...))"
        psams_test = np.ones((2, 10, 4))
        psams_test[0, 3:7] = np.eye(4) + 1e-10
        psams_test[1, 3:7] = np.eye(4)[::-1] + 1e-10
        to_search = np.random.RandomState(ap).randint(4, size=1000)
        motif = PSAMMotif.from_motifs(psams_test, ap, -1)
        sites = np.where(motif.score(to_search))[0].tolist()
        re_sites = [
            x.start() + motif.activation_point
            for x in re.finditer(regex, draw_bases(to_search))
        ]
        self.assertEqual(sites, re_sites)
