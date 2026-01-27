import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import line_search_utilities as lsu


def test_calculate_beta1_testcase1():
    alpha1 = 0.0
    alpha2 = 1.0
    phi_at_alpha1 = 1.0
    phi_at_alpha2 = 2.0
    dphi_at_alpha1 =-1.0
    dphi_at_alpha2 = 1.0
    
    assert lsu.calculate_beta1(alpha1,
                                alpha2,
                                phi_at_alpha1,
                                phi_at_alpha2,
                                dphi_at_alpha1,
                                dphi_at_alpha2) == pytest.approx(-3.0)

def test_calculate_beta2_testcase1():
    alpha1 = 0.0
    alpha2 = 1.0
    phi_at_alpha1 = 1.0
    phi_at_alpha2 = 2.0
    dphi_at_alpha1 =-1.0
    dphi_at_alpha2 = 1.0
    
    assert lsu.calculate_beta2(alpha1,
                                alpha2,
                                phi_at_alpha1,
                                phi_at_alpha2,
                                dphi_at_alpha1,
                                dphi_at_alpha2) == pytest.approx(np.sqrt(10))

def test_interpolate_minimum_alpha_testcase1():
    alpha1 = 0.0
    alpha2 = 1.0
    phi_at_alpha1 = 1.0
    phi_at_alpha2 = 2.0
    dphi_at_alpha1 =-1.0
    dphi_at_alpha2 = 1.0
    ans = 1 - (4 + np.sqrt(10))/(2 + 2*np.sqrt(10))
    
    assert lsu.interpolate_minimum_alpha(alpha1,
                                        alpha2,
                                        phi_at_alpha1,
                                        phi_at_alpha2,
                                        dphi_at_alpha1,
                                        dphi_at_alpha2) == pytest.approx(ans)
    

