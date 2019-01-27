'''
Question: 
    If we simulate multiplanet systems based on our findings from Kepler, how many yield an observable TTV?
    What about an observable RV signature? 

EPOS Simulation data:
    P: orbital period in days
    Y: planet radius in earth radii
    inc: inclination in degrees
    detectable: whether a simulated planet is detected (T/F)
    ID: stellar ID, to identify multi-planet systems

Steps:
    - Verfiy Nbody code can handle 3+ planets
    - Create Radius-Mass relation from existing data
    - Parse EPOS + Feed into Nbody code (check for unstable orbits?)
'''