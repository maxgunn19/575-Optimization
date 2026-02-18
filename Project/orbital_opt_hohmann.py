import numpy as np
from scipy.optimize import minimize

# ==========================================
# 1. CONSTANTS & MISSION PARAMETERS
# ==========================================
G0 = 9.81           # Gravity at sea level (m/s^2)
ISP = 450.0         # Specific Impulse (s) - e.g., LH2/LOX engine
DRY_MASS = 5000.0   # Structure + Payload (kg)
MIN_PAYLOAD = 1000.0 # Minimum payload requirement (kg) included in DRY_MASS here
MAX_G = 4.0         # Maximum G-force allowed

# Reference values for Normalization (To keep the optimizer balanced)
REF_FUEL = 20000.0  # kg
REF_TIME = 200.0    # days
REF_COST = 100.0    # Million $

# 3M Cost Factors (Hypothetical)
COST_PER_KG_FUEL = 0.005 # Million $ per kg
COST_PER_DAY_OPS = 0.1   # Million $ per day
FIXED_LAUNCH_COST = 50.0 # Million $

# ==========================================
# 2. PHYSICS MODELS (The "Black Box")
# ==========================================
def calculate_delta_v_required(time_of_flight_days):
    """
    Approximation of Lambert's Problem.
    Returns the Delta V (m/s) required to get to Mars given a specific flight time.
    In a high-fidelity model, this would solve the vector geometry.
    """
    # Simplified model: Faster flight = Exponentially more Delta V required
    # Hohmann transfer is approx 259 days with ~3600 m/s (LEO to Mars capture)
    # This curve approximates the penalty for "sprinting"
    
    optimal_time = 259.0 # Hohmann days
    min_dv = 3600.0      # m/s
    
    # Penalty factor for deviating from optimal time
    deviation = (optimal_time - time_of_flight_days)
    if deviation < 0: deviation = 0 # No penalty for going slower in this simple model
    
    # Simple exponential penalty model for demonstration
    dv_required = min_dv + 10 * (deviation**1.5) / 100
    return dv_required

def calculate_mission_cost(propellant_mass, time_of_flight):
    """Calculates cost in Millions of Dollars"""
    fuel_cost = propellant_mass * COST_PER_KG_FUEL
    ops_cost = time_of_flight * COST_PER_DAY_OPS
    return FIXED_LAUNCH_COST + fuel_cost + ops_cost

# ==========================================
# 3. OPTIMIZATION SETUP
# ==========================================

def objective_function(x, weights):
    """
    x[0]: Time of Flight (days)
    x[1]: Propellant Mass (kg)
    x[2]: Thrust Magnitude (Newtons)
    """
    tof, prop_mass, thrust = x
    w_fuel, w_time, w_cost = weights
    
    cost = calculate_mission_cost(prop_mass, tof)
    
    # Normalized Weighted Sum
    j_fuel = w_fuel * (prop_mass / REF_FUEL)
    j_time = w_time * (tof / REF_TIME)
    j_cost = w_cost * (cost / REF_COST)
    
    return j_fuel + j_time + j_cost

# ==========================================
# 4. CONSTRAINTS (Madsen's Focus)
# ==========================================
def constraint_delta_v(x):
    """
    Constraint: The rocket MUST provide enough Delta V to match the trajectory requirement.
    Rocket Eq: dV = Isp * g0 * ln(m_initial / m_final)
    """
    tof, prop_mass, thrust = x
    
    m_initial = DRY_MASS + prop_mass
    m_final = DRY_MASS
    
    dv_capacity = ISP * G0 * np.log(m_initial / m_final)
    dv_needed = calculate_delta_v_required(tof)
    
    # Result must be non-negative (Capacity - Needed >= 0)
    return dv_capacity - dv_needed

def constraint_g_loading(x):
    """
    Constraint: Max acceleration (at burnout) must be < MAX_G
    F = ma -> a = F/m
    """
    tof, prop_mass, thrust = x
    m_final = DRY_MASS # Lightest point
    accel_max = thrust / m_final
    g_load = accel_max / G0
    
    # Result must be non-negative (Max_G - Actual_G >= 0)
    return MAX_G - g_load

# ==========================================
# 5. RUNNER
# ==========================================
def run_optimization(scenario_name, weights):
    print(f"--- Running Scenario: {scenario_name} ---")
    print(f"Weights [Fuel, Time, Cost]: {weights}")
    
    # Initial Guesses [Days, kg Fuel, Newtons Thrust]
    x0 = [200.0, 15000.0, 50000.0]
    
    # Bounds for variables
    # (100 to 400 days), (1000kg to 100000kg fuel), (10kN to 200kN thrust)
    bnds = ((100, 400), (1000, 100000), (10000, 200000))
    
    # Define Constraints dictionary for SciPy
    cons = (
        {'type': 'ineq', 'fun': constraint_delta_v},
        {'type': 'ineq', 'fun': constraint_g_loading}
    )
    
    # Run Solver
    solution = minimize(objective_function, x0, args=(weights,), 
                        method='SLSQP', bounds=bnds, constraints=cons)
    
    if solution.success:
        t_opt, m_opt, f_opt = solution.x
        cost_opt = calculate_mission_cost(m_opt, t_opt)
        
        print(f"Optimization Successful!")
        print(f"  Flight Time:     {t_opt:.2f} days")
        print(f"  Propellant Mass: {m_opt:.2f} kg")
        print(f"  Thrust Force:    {f_opt:.2f} N")
        print(f"  Est. Total Cost: ${cost_opt:.2f} Million")
        print(f"  Delta V Capable: {ISP*G0*np.log((DRY_MASS+m_opt)/DRY_MASS):.2f} m/s")
        print(f"  Delta V Needed:  {calculate_delta_v_required(t_opt):.2f} m/s")
    else:
        print("Optimization Failed:", solution.message)
    print("\n")

# ==========================================
# 6. EXECUTE SCENARIOS
# ==========================================
if __name__ == "__main__":
    # Case 1: Minimize Fuel (High weight on fuel)
    run_optimization("Fuel Minimization", weights=[10.0, 1.0, 1.0])
    
    # Case 2: Sprint (High weight on time)
    run_optimization("Sprint (Time Min)", weights=[1.0, 10.0, 1.0])
    
    # Case 3: Balanced 3M Solution
    run_optimization("Balanced Mission", weights=[1.0, 1.0, 1.0])