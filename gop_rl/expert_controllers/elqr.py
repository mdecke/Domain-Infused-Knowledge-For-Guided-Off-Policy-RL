import numpy as np
import control

class LQRController:
    def __init__(self,
                 mass, rod_length, gravity, dt,
                 action_limits=(-2.0, 2.0),
                 Q = np.diag([50, 0.5]),
                 R = np.array([[0.2]])):
        """
        LQR Controller for the Pendulum environment.
        Parameters:
        mass: mass of the pendulum
        rod_length: length of the pendulum
        gravity: gravity
        action_limits: limits of the action space
        dt: time step
        Q: state cost matrix
        R: control effort cost
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.action_limits = action_limits
        self.dt = dt

        self.Q = Q
        self.R = R

        self.A = np.array([[0, 1],
                           [3 * self.g / (self.l * 2), 0.0]])
        self.B = np.array([[0], [1 / self.I]])

        self.K = control.lqr(self.A, self.B, self.Q, self.R)[0]

    def compute_control(self, state: np.ndarray, state_d: np.ndarray = np.array([0, 0])):
        """
        Compute the control action using the LQR feedback law.
        """
        u = - self.K @ (state - state_d)
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])

class EnergyShapingController:
    def __init__(self,
                 mass: float,
                 rod_length: float,
                 gravity: float,
                 dt: float,
                 action_limits: tuple = (-2.0, 2.0)):
        """
        Energy Shaping Controller for the Pendulum environment.
        See https://underactuated.mit.edu/acrobot.html#section6
        Parameters:
            mass: mass of the pendulum
            rod_length: length of the pendulum
            gravity: gravity
            action_limits: limits of the action space
            dt: time step
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.dt = dt
        self.action_limits = action_limits

    def compute_total_energy(self, state:np.ndarray):
        kinetic_energy = 1/2 * self.I * (state[1] ** 2)
        potential_energy = self.m * self.g * self.l/2 * np.cos(state[0])
        E = kinetic_energy + potential_energy
        return kinetic_energy, potential_energy, E

    def compute_desired_energy(self):
        E_d = self.m * self.g * self.l/2
        return E_d

    def get_action(self, state:np.ndarray, kp:float = 1.0):
        E_d = self.compute_desired_energy()
        _, _, E = self.compute_total_energy(state)

        E_err = E - E_d # Energy error.
        u = - kp * state[1] * E_err
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])