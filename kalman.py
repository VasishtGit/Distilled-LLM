class Kalman1D:
    def __init__(self, first_guess: float, guess_uncertainty: float = 1.0, drift: float = 1e-3, noise: float = 1e-1):
        """

        A simple Kalman filter for 1D numbers.

        - first_guess: our starting guess of the value
        - guess_uncertainty: how unsure we are about that guess
        - drift: how much we think the value itself can change over time
        - noise: how noisy we think the measurements are

        """
        self.best_value = first_guess        # our current best guess of the value
        self.uncertainty = guess_uncertainty # how much doubt we have in our guess
        self.drift = drift                   # how much the value drifts on its own
        self.noise = noise                   # how noisy the measurements are


    def update(self, new_measurement: float) -> float:

        """
        
        Update the guess when a new measurement comes in.

        """
        
        # We increase uncertainity as over time,before seeing a new measurement, we assume our guess is less certain
        self.uncertainty = self.uncertainty + self.drift

        # Kalman Gain  decides how much weight to give to the new measurement vs the old estimate
        kalman_gain = self.uncertainty / (self.uncertainty + self.noise)

        # Move the estimate toward the new measurement, controlled by K
        self.best_value = self.best_value + kalman_gain * (new_measurement - self.best_value)

        # After using the new measurement, uncertainty goes down
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

        # Return the new filtered estimate
        return self.best_value

