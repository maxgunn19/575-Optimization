import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta

# ==========================================
# 1. PHYSICS CONSTANTS & CONFIGURATION
# ==========================================
AU = 1.496e8  # km (Astronomical Unit)
MU_SUN = 1.327e11  # km^3/s^2 (Gravitational Parameter of Sun)

# Planet Data (Simplified Circular Model)
# Radius (km), Period (days), Distance from Sun (km)
EARTH_DIST = 1.0 * AU
MARS_DIST = 1.524 * AU
MOON_DIST_REAL = 384400  # km
MOON_DIST_VISUAL = 0.1 * AU # Exaggerated for visibility in plot!

EARTH_PERIOD = 365.25
MARS_PERIOD = 687.0
MOON_PERIOD = 27.3

# Optimization Window
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 12, 31)

# ==========================================
# 2. ORBITAL MECHANICS FUNCTIONS
# ==========================================

def get_planet_pos(radius, period, t_days, phase_offset=0):
    """Calculates x, y position of a planet at time t."""
    # Mean motion (rad/day)
    n = 2 * np.pi / period
    theta = n * t_days + phase_offset
    return radius * np.cos(theta), radius * np.sin(theta)

def solve_hohmann_transfer(r1, r2):
    """Calculates Hohmann transfer properties."""
    # Semi-major axis of transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Time of Flight (seconds -> days)
    t_flight_sec = np.pi * np.sqrt(a_transfer**3 / MU_SUN)
    t_flight_days = t_flight_sec / (24 * 3600)
    
    # Required Phase Angle at launch (radians)
    # How far Mars moves during the transfer
    n_mars = 2 * np.pi / MARS_PERIOD
    mars_travel_angle = n_mars * t_flight_days
    
    # For arrival, Mars must be at pi radians (180 deg) from launch longitude + travel
    # Initial separation required:
    required_phase = np.pi - mars_travel_angle
    
    return t_flight_days, required_phase

# ==========================================
# 3. OPTIMIZER (Find Best Launch Date)
# ==========================================

def optimize_launch_date():
    """Finds the date between 2020-2025 where Earth/Mars phase angle is optimal."""
    print("Optimizing Launch Window...")
    
    # Ideal phase angle for Earth->Mars
    _, target_phase = solve_hohmann_transfer(EARTH_DIST, MARS_DIST)
    
    best_date_days = 0
    min_error = float('inf')
    
    total_days = (END_DATE - START_DATE).days
    
    # Check every day in the window
    for day in range(total_days):
        # Calculate angular positions
        theta_earth = (2 * np.pi / EARTH_PERIOD) * day
        theta_mars = (2 * np.pi / MARS_PERIOD) * day
        
        # Current phase difference (normalized to -pi to pi)
        current_phase = (theta_mars - theta_earth) % (2 * np.pi)
        if current_phase > np.pi: current_phase -= 2*np.pi
        
        # Error from ideal
        error = abs(current_phase - target_phase)
        
        if error < min_error:
            min_error = error
            best_date_days = day
            
    launch_date_obj = START_DATE + timedelta(days=best_date_days)
    print(f"Optimal Launch Found: {launch_date_obj.strftime('%Y-%m-%d')}")
    print(f"Launch Day Index: {best_date_days}")
    return best_date_days

# ==========================================
# 4. SIMULATION & ANIMATION
# ==========================================

def run_simulation():
    # 1. Setup Optimization
    launch_day = optimize_launch_date()
    tof, _ = solve_hohmann_transfer(EARTH_DIST, MARS_DIST)
    arrival_day = launch_day + tof
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    
    # Plot Limits (Ensure everything fits)
    limit = 1.8 * AU
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    # Static Elements (Orbits)
    earth_orbit = plt.Circle((0, 0), EARTH_DIST, color='blue', linestyle='--', fill=False, alpha=0.3, label='Earth Orbit')
    mars_orbit = plt.Circle((0, 0), MARS_DIST, color='green', linestyle='--', fill=False, alpha=0.3, label='Mars Orbit')
    sun = plt.Circle((0, 0), 0.05*AU, color='yellow', label='Sun')
    ax.add_patch(earth_orbit)
    ax.add_patch(mars_orbit)
    ax.add_patch(sun)
    
    # Dynamic Elements (Dots)
    earth_dot, = ax.plot([], [], 'bo', markersize=8, label='Earth')
    mars_dot, = ax.plot([], [], 'go', markersize=8, label='Mars')
    moon_dot, = ax.plot([], [], 'w.', markersize=4) # Moon is white dot
    rocket_dot, = ax.plot([], [], 'r*', markersize=10, label='Rocket')
    rocket_trail, = ax.plot([], [], 'r--', linewidth=1, alpha=0.7)
    
    # Info Text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')
    status_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='lime')
    
    ax.legend(loc='lower right')

    # Trail history
    trail_x, trail_y = [], []

    def init():
        earth_dot.set_data([], [])
        mars_dot.set_data([], [])
        moon_dot.set_data([], [])
        rocket_dot.set_data([], [])
        rocket_trail.set_data([], [])
        time_text.set_text('')
        status_text.set_text('')
        return earth_dot, mars_dot, moon_dot, rocket_dot, rocket_trail, time_text, status_text

    def update(frame):
        # Frame is current day relative to simulation start (0)
        # We start animation 100 days before launch to show alignment
        current_day = (launch_day - 100) + frame
        
        # 1. Update Planets
        ex, ey = get_planet_pos(EARTH_DIST, EARTH_PERIOD, current_day)
        mx, my = get_planet_pos(MARS_DIST, MARS_PERIOD, current_day)
        
        # Update Moon (Relative to Earth)
        # Note: Speed is exaggerated x5 so you can see it orbit in the short timeframe
        # Distance is exaggerated so it's not inside the Earth dot
        moon_x_rel, moon_y_rel = get_planet_pos(MOON_DIST_VISUAL, MOON_PERIOD, current_day)
        moon_x, moon_y = ex + moon_x_rel, ey + moon_y_rel
        
        earth_dot.set_data([ex], [ey])
        mars_dot.set_data([mx], [my])
        moon_dot.set_data([moon_x], [moon_y])
        
        # 2. Update Rocket
        if current_day < launch_day:
            # Pre-launch: Rocket is on Earth
            rx, ry = ex, ey
            status = "Status: Awaiting Launch Window"
        elif current_day <= arrival_day:
            # In Transit: Calculate Transfer Orbit Position
            # Progress (0.0 to 1.0)
            progress = (current_day - launch_day) / tof
            
            # Transfer Orbit Physics (Polar coordinates)
            # Angle moves from Earth's launch angle to Mars' arrival angle (approx pi radians sweep)
            
            # Start angle (Earth at launch)
            theta_start = (2 * np.pi / EARTH_PERIOD) * launch_day
            
            # Current angle in transfer (sweeping 180 degrees / pi radians)
            theta_transfer = theta_start + (progress * np.pi)
            
            # Radius (Polar equation of ellipse, focus at Sun)
            # r = p / (1 + e cos(nu))
            # For Hohmann, r varies from r1 to r2
            # Simplified visualization interpolation:
            r_transfer = EARTH_DIST + (MARS_DIST - EARTH_DIST) * progress 
            # (Note: Linear radius interp is a visual approximation for smooth animation 
            # without solving Kepler's eqn every frame)
            
            rx = r_transfer * np.cos(theta_transfer)
            ry = r_transfer * np.sin(theta_transfer)
            
            trail_x.append(rx)
            trail_y.append(ry)
            status = "Status: TRANSIT TO MARS"
        else:
            # Arrival: Rocket stays on Mars
            rx, ry = mx, my
            status = "Status: MISSION COMPLETE"
            
        rocket_dot.set_data([rx], [ry])
        rocket_trail.set_data(trail_x, trail_y)
        
        # Update Text
        current_date = START_DATE + timedelta(days=current_day)
        time_text.set_text(f"Date: {current_date.strftime('%Y-%m-%d')}")
        status_text.set_text(status)
        
        return earth_dot, mars_dot, moon_dot, rocket_dot, rocket_trail, time_text, status_text

    # Total frames: 100 days pre-launch + TOF + 100 days post-arrival
    total_frames = int(100 + tof + 100)
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                  init_func=init, blit=True, interval=20)
    
    plt.title(f"3M Aerospace: Optimal Earth-Mars Transfer\nLaunch: {START_DATE + timedelta(days=launch_day):%Y-%m-%d}")
    plt.show()

if __name__ == "__main__":
    run_simulation()