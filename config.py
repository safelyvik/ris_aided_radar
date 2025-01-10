# config.py
# Configuration file for RIS optimization

class Config:
    # Radar and RIS parameters
    NUM_RIS_ELEMENTS = 64
    PHASE_SHIFTS = 16  # Discrete values for phase shifts
    SIGNAL_FREQUENCY = 3e9  # Hz
    BANDWIDTH = 50e6  # Hz
    PULSE_DURATION = 1e-6  # seconds

    # Machine learning parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    BATCH_SIZE = 32

    # Data generation
    NUM_TRAIN_SAMPLES = 10000
    NUM_TEST_SAMPLES = 2000

    # Environment setup
    RADAR_DISTANCE = 100  # meters
    TARGET_DISTANCE = 5000  # meters
    CLUTTER_COUNT = 20

# Example: Access constants using Config.NUM_RIS_ELEMENTS
