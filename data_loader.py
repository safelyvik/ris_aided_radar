import numpy as np

def generate_synthetic_data(num_targets, num_scatterers, ris_elements, noise_power):
    """
    Generate synthetic data for training or testing the RIS optimization system.

    Args:
        num_targets (int): Number of targets in the simulation.
        num_scatterers (int): Number of clutter scatterers in the environment.
        ris_elements (int): Number of RIS elements.
        noise_power (float): Power of additive noise.

    Returns:
        dict: Dictionary containing the radar returns, clutter, and noise.
    """
    # Generate random target returns
    target_amplitudes = np.random.uniform(1, 10, size=num_targets)
    target_phases = np.random.uniform(0, 2 * np.pi, size=num_targets)
    target_returns = target_amplitudes * np.exp(1j * target_phases)

    # Generate random scatterer returns
    scatterer_amplitudes = np.random.uniform(0.5, 5, size=num_scatterers)
    scatterer_phases = np.random.uniform(0, 2 * np.pi, size=num_scatterers)
    scatterer_returns = scatterer_amplitudes * np.exp(1j * scatterer_phases)

    # Generate RIS phase matrix (random initial values)
    ris_phases = np.random.uniform(0, 2 * np.pi, size=(ris_elements,))

    # Add noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_targets + num_scatterers) + 1j * np.random.randn(num_targets + num_scatterers))

    return {
        "targets": target_returns,
        "scatterers": scatterer_returns,
        "ris_phases": ris_phases,
        "noise": noise,
    }

def load_data():
    """
    Load or generate data for RIS optimization.

    Returns:
        dict: Dataset for RIS optimization.
    """
    from config import NUM_TARGETS, NUM_SCATTERERS, RIS_ELEMENTS, NOISE_POWER
    return generate_synthetic_data(NUM_TARGETS, NUM_SCATTERERS, RIS_ELEMENTS, NOISE_POWER)

if __name__ == "__main__":
    data = load_data()
    print("Generated Data:")
    for key, value in data.items():
        print(f"{key}: {value[:5] if isinstance(value, np.ndarray) else value}")
