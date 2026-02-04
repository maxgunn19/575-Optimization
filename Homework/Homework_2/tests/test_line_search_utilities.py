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
    

# --- EXTRA TEST CASE 1 --- #
def test_random_case_2():
    # Function: f(x) = (x - 2)^2 
    a1, a2 = 1.0, 3.0
    p1, p2 = 1.0, 1.0       # f(1)=1, f(3)=1
    dp1, dp2 = -2.0, 2.0    # f'(1)=-2, f'(3)=2

    # Hand Calculations (Python does the math for us to generate the "answer key")
    # Beta1 = dp1 + dp2 - 3*(p1 - p2)/(a1 - a2)
    b1_expected = 0.0
    
    # Beta2 = sqrt(b1^2 - dp1*dp2)  (since a2 > a1)
    b2_expected = 2.0
    
    # Alpha = a2 - (a2 - a1) * (dp2 + b2 - b1) / (dp2 - dp1 + 2*b2)
    alpha_expected = 2.0

    # Assertions
    assert lsu.calculate_beta1(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b1_expected)
    assert lsu.calculate_beta2(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b2_expected)
    assert lsu.interpolate_minimum_alpha(a1, a2, p1, p2, dp1, dp2) == pytest.approx(alpha_expected)


    # --- EXTRA TEST CASE 2 --- #
def test_random_case_3():
    # Function: f(x) = 1/3*x^3 - 2x^2 + 3x
    a1, a2 = 2.0, 4.0
    p1, p2 = 2.0/3.0, 4.0/3.0
    dp1, dp2 = -1.0, 3.0
    
    # Hand Calculations (Python does the math for us to generate the "answer key")
    # Beta1 = dp1 + dp2 - 3*(p1 - p2)/(a1 - a2)
    b1_expected = 1.0
    
    # Beta2 = sqrt(b1^2 - dp1*dp2)  (since a2 > a1)
    b2_expected = 2.0
    
    # Alpha = a2 - (a2 - a1) * (dp2 + b2 - b1) / (dp2 - dp1 + 2*b2)
    alpha_expected = 3.0

    # Assertions
    assert lsu.calculate_beta1(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b1_expected)
    assert lsu.calculate_beta2(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b2_expected)
    assert lsu.interpolate_minimum_alpha(a1, a2, p1, p2, dp1, dp2) == pytest.approx(alpha_expected)

    # --- EXTRA TEST CASE 3 --- #
def test_random_case_4():
    # Function: f(x) = x^2
    a1, a2 = -1.0, 2.0
    p1, p2 = 1.0, 4.0
    dp1, dp2 = -2.0, 4.0
    
    # Hand Calculations (Python does the math for us to generate the "answer key")
    # Beta1 = dp1 + dp2 - 3*(p1 - p2)/(a1 - a2)
    b1_expected = -1.0
    
    # Beta2 = sqrt(b1^2 - dp1*dp2)  (since a2 > a1)
    b2_expected = 3.0
    
    # Alpha = a2 - (a2 - a1) * (dp2 + b2 - b1) / (dp2 - dp1 + 2*b2)
    alpha_expected = 0.0

    # Assertions
    assert lsu.calculate_beta1(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b1_expected)
    assert lsu.calculate_beta2(a1, a2, p1, p2, dp1, dp2) == pytest.approx(b2_expected)
    assert lsu.interpolate_minimum_alpha(a1, a2, p1, p2, dp1, dp2) == pytest.approx(alpha_expected)