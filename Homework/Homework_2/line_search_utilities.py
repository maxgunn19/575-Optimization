import numpy as np


def calculate_beta1(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2):
    beta1 = dphi_at_alpha1 + dphi_at_alpha2 - (3 * (phi_at_alpha2 - phi_at_alpha1) / (alpha_2 - alpha_1))
    return beta1

def calculate_beta2(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2):
    beta1 = calculate_beta1(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2)
    
    beta2 = np.sign(alpha_2 - alpha_1) * np.sqrt(beta1**2 - dphi_at_alpha1 * dphi_at_alpha2)
    return beta2

def interpolate_minimum_alpha(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2):
    beta1 = calculate_beta1(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2)
    beta2 = calculate_beta2(alpha_1, alpha_2, phi_at_alpha1, phi_at_alpha2, dphi_at_alpha1, dphi_at_alpha2)
    alpha_min = alpha_2 - (alpha_2 - alpha_1) * (dphi_at_alpha2 + beta2 - beta1) / (dphi_at_alpha2 - dphi_at_alpha1 + 2 * beta2)
    return alpha_min