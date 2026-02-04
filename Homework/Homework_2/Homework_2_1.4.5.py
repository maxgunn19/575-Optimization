import numpy as np

def calculate_beta1(a1, a2, phi1, phi2, dphi1, dphi2):
    beta1 = dphi1 + dphi2 - 3 * (phi1 - phi2) / (a1 - a2)
    
    return beta1


def calculate_beta2(a1, a2, phi1, phi2, dphi1, dphi2, beta1):    
    beta2 = np.sign(a2 - a1) * np.sqrt(max(0, beta1**2 - dphi1 * dphi2))
    
    return beta2


def interpolate_minimum_alpha(a1, a2, phi1, phi2, dphi1, dphi2):    
    b1 = calculate_beta1(a1, a2, phi1, phi2, dphi1, dphi2)
    b2 = calculate_beta2(a1, a2, phi1, phi2, dphi1, dphi2, b1)
    
    numerator = dphi2 + b2 - b1
    denominator = dphi2 - dphi1 + 2 * b2
    
    if abs(denominator) < 1e-12:
        return a1 + (a2 - a1) / 2.0
        
    alpha_star = a2 - (a2 - a1) * (numerator / denominator)
    
    return alpha_star


def pinpoint(a_low, a_high, phi_0, dphi_0, phi_low, dphi_low, mu1, mu2, func_phi):    
    while True:
        phi_high, dphi_high = func_phi(a_high)
        
        ap = interpolate_minimum_alpha(a_low, a_high, phi_low, phi_high, dphi_low, dphi_high)
        phi_p, dphi_p = func_phi(ap)

        if phi_p > (phi_0 + mu1 * ap * dphi_0) or phi_p >= phi_low:
            a_high = ap
        else:
            if abs(dphi_p) <= -mu2 * dphi_0:
                return ap
                
            if dphi_p * (a_high - a_low) >= 0:
                a_high = a_low

            a_low = ap
            phi_low, dphi_low = phi_p, dphi_p


def line_search(x, p, fun, grad, mu1=1e-4, mu2=0.1, sigma=2.0):    
    def phi_alpha(alpha):
        x_curr = x + alpha * p
        f_val = fun(x_curr)
        g_val = grad(x_curr)
        return f_val, np.dot(g_val, p)
    
    a1 = 0.0
    a2 = 1.0
    
    phi_0, dphi_0 = phi_alpha(0.0)
    phi1, dphi1 = phi_0, dphi_0
    
    first_iteration = True

    while True:
        phi2, dphi2 = phi_alpha(a2)

        if (phi2 > phi_0 + mu1 * a2 * dphi_0) or (not first_iteration and phi2 >= phi1):
            return pinpoint(a1, a2, phi_0, dphi_0, phi1, dphi1, mu1, mu2, phi_alpha)
        
        if abs(dphi2) <= -mu2 * dphi_0:
            return a2
        
        if dphi2 >= 0:
            return pinpoint(a2, a1, phi_0, dphi_0, phi2, dphi2, mu1, mu2, phi_alpha)
        
        a1, phi1, dphi1 = a2, phi2, dphi2
        a2 = sigma * a2
        first_iteration = False


def slanted_quadratic(x, beta=1.5):    
    quad_out = x[0]**2 + x[1]**2 - beta * x[0] * x[1]
    
    return quad_out


def grad_slanted_quadratic(x, beta=1.5):    
    df1 = 2.0 * x[0] - beta * x[1]
    df2 = 2.0 * x[1] - beta * x[0]
    
    return np.array([df1, df2])


def rosenbrock(x):   
    ros_out = (1 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
    return ros_out


def grad_rosenbrock(x):    
    df1 = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0])
    df2 = 200.0 * (x[1] - x[0]**2)
    
    return np.array([df1, df2])


def jones(x):    
    jones_out = x[0]**4 + x[1]**4 - 4.0 * x[0]**3 - 3.0 * x[1]**3 + 2 * x[0]**2 + 2 * x[0] * x[1]
    
    return jones_out


def grad_jones(x):    
    df1 = 4.0 * x[0]**3 - 12.0 * x[0]**2 + 4.0 * x[0] + 2.0 * x[1]
    df2 = 4.0 * x[1]**3 - 9.0 * x[1]**2 + 2.0 * x[0]
    
    return np.array([df1, df2])


call_count = 0

def wrap_func(func, x):    
    global call_count
    call_count += 1
    
    return func(x)


def run_problem_1_5_tests():    
    global call_count
    
    params = {
        'mu1': 1e-4, 
        'mu2': 0.1, 
        'sigma': 2.0
    }

    test_cases = [
        ("Slanted Quadratic", slanted_quadratic, grad_slanted_quadratic, 
         np.array([2.0, -6.0]), np.array([-1.0, 1.0])),
        ("Rosenbrock", rosenbrock, grad_rosenbrock, 
         np.array([0.0, 2.0]), np.array([1.0, -3.0])),
        ("Jones", jones, grad_jones, 
         np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    ]

    for name, func, grad, x0, p in test_cases:
        call_count = 0

        def counted_func(x): 
            return wrap_func(func, x)

        alpha_star = line_search(x0, p, counted_func, grad, **params)
        x_final = x0 + alpha_star * p

        print(f"Results for {name}:")
        print(f"  Optimal Step Length (alpha*): {alpha_star}")
        print(f"  Point Found (x_final): {x_final}")
        print(f"  Function calls: {call_count}")
        print("")


if __name__ == "__main__":
    run_problem_1_5_tests()