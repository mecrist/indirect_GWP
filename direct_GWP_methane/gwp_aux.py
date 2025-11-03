import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A
from math import exp

###################### CONVOLUTED ESPECTRA #########################
def read_orca(path_out):
    """
    Extracts anharmonic IR frequencies (cm^-1) and intensities (km/mol)
    from the 'IR SPECTRUM' section of an ORCA output
    """
    freqs, intensities = [], []

    with open(path_out, 'r') as f:
        lines = f.readlines()

    found_section = False
    for line in lines:
        if 'IR SPECTRUM' in line:
            found_section = True
            continue
        if found_section:
            #skip blank or separator lines
            if line.strip() == '' or '-----' in line:
                continue
            # extract (freq, intensity) from lines like "10:  676.63 ... 130.70 ..."
            match = re.match(r'\s*\d+:\s*([\d.]+)\s+[.\dE+-]+\s+([\d.]+)', line)
            if match:
                freqs.append(float(match.group(1)))
                intensities.append(float(match.group(2)))

    return np.array(freqs), np.array(intensities)

def lorentzian(x, x0, gamma, intensity):
    """
    Lorentzian broadening function.
    The integral of this function over all x equals the intensity.
    
    Parameters:
    x: frequency grid
    x0: center frequency
    gamma: half-width at half-maximum (HWHM)
    intensity: total integrated intensity
    """
    return (intensity * gamma / np.pi) / ((x - x0)**2 + gamma**2)

def convolute_ir_spectrum(freqs, intensities, x_range=(0, 3000), hwhm=30.0, n_points=30000):
    """
    Convolute stick IR spectrum with Lorentzian broadening.
    
    Parameters:
    freqs: array of frequencies (cm^-1)
    intensities: array of intensities (km/mol)
    x_range: tuple (min_freq, max_freq) for the output spectrum
    hwhm: half-width at half-maximum for Lorentzian (cm^-1)
    n_points: number of points in the output spectrum
    
    Returns:
    x: frequency grid
    y_convoluted: convoluted spectrum
    """
    
    # Create frequency grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y_convoluted = np.zeros_like(x)
    
    # Convolute each peak with Lorentzian
    for freq, intensity in zip(freqs, intensities):
        y_convoluted += lorentzian(x, freq, hwhm, intensity)  
    
    return x, y_convoluted

def plot_ir_spectrum(path_out, hwhm=30.0, x_range=(0,3000), figsize=(10, 6)):
    freqs, intensities = read_orca(path_out)
    
    x, y_convoluted = convolute_ir_spectrum(freqs, intensities, x_range, hwhm)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y_convoluted, 'b-', linewidth=1.5, label=f'Convoluted (HWHM={hwhm} cm⁻¹)')
    
    # Plot stick spectrum
    ax.vlines(freqs, 0, intensities, colors='r', linewidth=1.5, alpha=0.7, label='Stick spectrum')
    
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (km/mol)')
    ax.set_title('IR Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if x_range:
        ax.set_xlim(x_range)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, freqs, intensities, x, y_convoluted

################## unit conversion ####################
def convert_kmpermol_to_cmpermolecule(intens_convoluted):
    intensities_scaled = []
    for i in intens_convoluted:
        i_scaled = (i * 1e5)/N_A
        intensities_scaled.append(i_scaled)

    return intensities_scaled

##########################AVERAGE MEAN CROSS SECTION#############################
def mean_cross_section(intensities_scaled, freqs, delta_nu=1):
    """
    Calculate mean intensities in specified wavenumber intervals.
    
    Parameters:
    intens_convoluted (list/array): List of intensity values
    freqs (list/array): List of frequency/wavenumber values
    delta_nu (float): Interval width in cm⁻¹ (default: 1)
    
    Returns:
    list: Averaged intensities for each interval
    """
    
    # Convert inputs to numpy arrays
    freqs = np.array(freqs)
    intensities_scaled = np.array(intensities_scaled)
    
    min_freq = 0
    max_freq = 3000
    
    # Create bins for the intervals of frequencies
    bins = np.arange(min_freq, max_freq + delta_nu, delta_nu)

    # Filter frequencies and intensities that fall within the 0-2500 range
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    filtered_freqs = freqs[mask]
    filtered_intens = intensities_scaled[mask]
    
    # Digitize the filtered frequencies to assign them to bins
    bin_indices = np.digitize(filtered_freqs, bins) - 1
    
    # Calculate mean intensity for each bin (3000 bins total)
    averaged_intensities = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.any(mask):
            mean_intensity = np.mean(filtered_intens[mask])
            averaged_intensities.append(mean_intensity)
        else:
            averaged_intensities.append(0.0)
    
    return averaged_intensities

################RADIATIVE EFICIENCY CALCULATION##################

def lifetime_correction_factor(tau, degradation_type='photolysis'):
    """
    Apply lifetime correction factor to radiative efficiency (RE)
    
    Parameters:
    tau (float): Atmospheric lifetime in years
    degradation_type (str): 'photolysis' or 'OH' (default)
    
    Returns:
    float: Correction factor f(τ)
    """
    if degradation_type == 'photolysis':
        # For gases degraded by photolysis (10 < τ < 10^4 years)
        if not (10 < tau < 1e4):
            print(f"Warning: τ={tau} outside recommended range (10, 10000) for photolysis")
        return 1 - 0.1826 * (tau ** -0.3339)
    
    elif degradation_type == 'OH':
        # For gases that react with OH (10^-4 < τ < 10^4 years)
        if not (1e-4 < tau < 1e4):
            print(f"Warning: τ={tau} outside recommended range (0.0001, 10000) for OH reaction")
        
        a = 2.962
        b = 0.9312
        c = 2.994
        d = 0.9302
        
        numerator = a * (tau ** b)
        denominator = 1 + c * (tau ** d)
        return numerator / denominator
    
    else:
        raise ValueError("degradation_type must be 'photolysis' or 'OH'")

def calculate_radiative_efficiency(averaged_intensities, F_converted, tau, degradation_type):
    """
    Calculates the radiative efficiency (RE) in W m⁻²/ppbv.
    
    Args:
        averaged_intensities: Mean cross-section intensities for each bin
        F_converted: Instantaneous radiative forcing array from table (mW m⁻²)
        tau: atmospheric lifetime in years
        degradation_type: 'photolysis' or 'OH'
    
    Returns:
        float: Radiative efficiency with stratospheric‑temperature adjustment.  
    """
    
    avg_IRE = []
    
    for cs, conv in zip(averaged_intensities, F_converted):
        IRE = cs * conv 
        avg_IRE.append(IRE)
        
    RE = (sum(avg_IRE))
    
    #stratospheric‑temperature adjustment (Shine and Myhre,2020)
    f_tau = lifetime_correction_factor(tau, degradation_type)
    RE_STA = RE * f_tau
    
    return RE_STA

###############UNIT CONVERSION################
def convert_ppbv_to_kg(RE, molecular_weight):
    T_m = 5.148e18 #kg, massa total da atm
    M_atm = 28.96 ## km/mol, peso molecular médio do ar

    n_air_total = M_atm / molecular_weight

    # 2) mols de gás para 1 ppb global
    n_gas_ppb = 1e9 / T_m

    RE_kg = RE * n_air_total * n_gas_ppb

    # 4) converte RE para W/m²/kg
    return RE_kg


############################# GWP CALCULATIONN##################################
def AGWP_gas(RE, tau, H):
    return RE * tau * (1 - exp(-H / tau))

def calculate_gwp(RE_W_m2_ppb, tau, molar_mass, horizons=[20, 100, 500]):
    AGWP_co2 = [0.0243, 0.0895, 0.314] #AR6 IPCC VALUES - mudei aqui
    
    # Converte eficiência radiativa de W/m²/ppb para W/m²/kg
    RE_corr_kg = convert_ppbv_to_kg(RE_W_m2_ppb, molar_mass) 
    
    # Converte para pW/m²/kg (1 W = 1e12 pW) para melhor precisão numérica
    RE_corr_pW = RE_corr_kg * 1e12
    
    results = {}
    
    # Calcula para cada horizonte temporal
    for H, agwp in zip(horizons, AGWP_co2):
        # Integra AGWP para o gás (pW·ano/m² por kg de gás)
        AGWP_g = AGWP_gas(RE_corr_pW, tau, H)
        
        # Calcula GWP como a razão entre os AGWPs
        GWP = AGWP_g / agwp
        
        # Armazena resultados
        results[f'GWP_{H}yr'] = GWP
        results[f'AGWP_gas_{H}yr'] = AGWP_g    
    return results
