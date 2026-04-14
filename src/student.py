import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng()

from src.utils import (
    is_number, 
    is_singular, 
    integral, 
    derivative, 
    lu_decomposition,
)


class TUDStudent:
    """Represents a student at the Technical University of Darmstadt."""
    def __init__ (self, name, age, date_of_registration, registration_number, 
                  finished_courses, favorite_course):
        self._name = name
        self._age = age
        self._date_of_registration = date_of_registration
        self._registration_number = registration_number
        self.finished_courses = finished_courses
        self.favorite_course = favorite_course
        
    @property
    def name(self):
        return self._name
    
    @property
    def age(self):
        return self._age
    
    @property
    def date_of_registration(self):
        return self._date_of_registration
    
    @property
    def registration_number(self):
        return self._registration_number
    
    @property
    def finished_courses(self):
        return self._finished_courses
    
    @finished_courses.setter
    def finished_courses(self, courses):
        if not isinstance(courses, list):
            raise TypeError("Courses must be a list.")
        if not all(isinstance(course, str) for course in courses):
            raise TypeError("All values in courses must be strings.")
        self._finished_courses = courses
        
    @property
    def favorite_course(self):
        return self._favorite_course
    
    @favorite_course.setter
    def favorite_course(self, course):
        if not isinstance(course, str):
            raise TypeError("Favorite course must be a string.")
        self._favorite_course = course
        
                
class DataScienceStudent(TUDStudent):
    """
    Represents a data science student at the Technical University of Darmstadt.
    Inherits from TUDStudent and adds methods for solving math problems.
    """
    def solve_integral(self, x_limits, x_stats, f=lambda x: np.exp(-x) * np.cos(x)):
        try:
            x_start, x_end = x_limits
            if not (is_number(x_start) and is_number(x_end)):
                raise TypeError 
        except (TypeError, ValueError): 
            raise TypeError("x_limits must be a Sequence of two numbers.")
        try:
            a, b = x_stats
            if not (is_number(a) and is_number(b)):
                raise TypeError
        except (TypeError, ValueError): 
            raise TypeError("x_stats must be a Sequence of two numbers.")    
        
        # 1. Compute y and plot the function
        x = np.linspace(x_start, x_end, int(np.ceil(1000*(x_end - x_start))))
        y = f(x)
        
        # 2. Compute mean, variance, and standard deviation
        # Integrate f(x) over [a, b] using midpoint Riemann sum
        Y_ab, y_ab, x_ab, dx = integral(f, a, b, n=int(np.ceil(1000*(b - a))))
        
        mean = Y_ab / (b - a)
        variance_temp, _, _, _ = integral(lambda x: (f(x) - mean) ** 2, a, b)
        variance = variance_temp / (b - a)
        sd = np.sqrt(variance)
        
        # 3. Find the threshold 70% of y is below
        y_sorted = np.sort(y_ab)
        thresh_idx = int(0.7 * len(y_sorted))
        y_m = y_sorted[thresh_idx]
        
        # 4. Compute derivative
        delta_x = x[1] - x[0]
        dydx = derivative(delta_x, y)
        
        # 5. Find extrema
        # Extremum are identified by an exact zero crossing or a sign change
        # between two consecutive points in the derivative
        extrema_idx = []
        for i in range(1, len(dydx) - 1):
            if dydx[i] == 0 or dydx[i-1] * dydx[i] < 0:
                extrema_idx.append(i + 1)
                
        extrema_x = x[extrema_idx]
        extrema_y = y[extrema_idx]
        
        # 6. Console output
        print(
            f"Mean                  = {mean:.4g} \n"
            f"Variance              = {variance:.4g} \n"
            f"Standard Deviation    = {sd:.4g} \n"
            f"70% threshold         = {y_m:.4g} \n"
            "\n"
            f"Extrema at: \n" 
            + "\n".join(f"x_{i} = {x:.4g}, " for i, x in enumerate(extrema_x))
        )
        
        return {
            "x": x,
            "y": y,
            "x_interval": x_ab,
            "y_interval": y_ab,
            "mean": mean,
            "variance": variance,
            "standard_deviation": sd,
            "threshold_70_percent": y_m,
            "dx": dx,
            "dydx": dydx,
            "extrema_index": extrema_idx,
            "extrema_x": extrema_x
        }
        
    def plot_integral(
        self, x_limits, 
        x_stats, 
        f=lambda x: np.exp(-x) * np.cos(x), 
        derivative=True
    ):
        """
        Helper method to plot the data from solve_integral. This allows 
        solve_integral to return the results without plotting.
        """
        results = self.solve_integral(x_limits, x_stats, f)

        fig, ax = plt.subplots(layout="constrained")
        ax.plot(results["x"], results["y"], label=r"$y = f(x)$")
        ax.bar(
            results["x_interval"], 
            results["y_interval"], 
            width=results["dx"], 
            color="gray", 
            alpha=0.5, 
            ec="gray",
            label=f"Area under y on [{x_stats[0]}, {x_stats[1]}]"
        )
        if derivative:
            plt.plot(results["x"][1:-1], results["dydx"], label=r"$\frac{dy}{dx}$")
            plt.scatter(
                results["x"][results["extrema_index"]], 
                results["y"][results["extrema_index"]], 
                color="red",
                label=r"$\frac{dy}{dx} = 0$"
            )
        plt.legend()
        plt.show()   

    def solve_SLE(self, A, b):
        # Check input type by converting to numpy array
        try:
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except Exception:
            raise TypeError(
                "Input must be convertible to a numpy array of numbers."
            )
        
        if is_singular(A):
            raise ValueError("Coefficient matrix A is singular.")
        
        L, U = lu_decomposition(A)
        
        # Forward substitution to solve Ly = b
        y = np.zeros_like(b, dtype=float)
        
        for i in np.arange(L.shape[0]):
            sum = 0
            for j in np.arange(i):
                sum += y[j] * L[i, j]
                
            y[i] = (b[i] - sum) / L[i, i]
        
        # Backward substitution to solve Ux = y
        x = np.zeros_like(y, dtype=float)
        
        for i in np.arange(U.shape[0] - 1, -1, -1):
            sum = 0
            for j in np.arange(i + 1, U.shape[0]):
                sum += x[j] * U[i, j]
                
            x[i] = (y[i] - sum) / U[i, i]
        
        print(f"Solution to SLE: {x}")
        
        return x
    
    def invert_matrix(self, A):
        if A.size == 1:
            if A == 0:
                raise ValueError("Cannot invert a zero scalar.")
            return 1.0 / A
    
        # Column-wise inversion by solving SLE for each column
        inv_A = np.zeros(A.shape)
        e = np.eye(len(A))
        
        for i in np.arange(len(A)):
            inv_A[:, i] = self.solve_SLE(A, e[:, i])
        
        return inv_A
        
    def solve_OLS(self, y, X, output=True):
        # Check input type by converting to numpy array
        try:
            y = np.array(y, dtype=float)
            X = np.array(X, dtype=float)
        except Exception:
            raise TypeError(
                "Input must be convertible to a numpy array of numbers."
            )
        
        # Handle 1d input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check restriction observations > variables
        n, k = X.shape
        if n < k:
            raise ValueError("Number of observations must exceed number of variables.")
        
        # (X.T @ X)^-1 is used twice: calculating parameters beta and 
        # the covariance matrix C
        xtx_inv = self.invert_matrix(X.T @ X)
        
        # Solve for model parameter vector beta
        beta = xtx_inv @ X.T @ y

        # Calculating standard errors for t-statistics
        residuals = y - X @ beta
        sigma_squared = sum(residuals ** 2) / (n - k)
        cov_matrix = sigma_squared * xtx_inv
        standard_errors = np.sqrt(np.diag(cov_matrix))
        
        # Calculate t-statistics and p-values
        t_stats = beta / standard_errors
        p_values = 2 * stats.t.sf(np.abs(t_stats), n - k)
        
        # Optional console output
        if output:
            print(f"Parameter vector beta:  {beta} \n")
            print(f"t-statistics:           {t_stats} \n")
            print(f"p-values:               {p_values}")
            
        return beta, t_stats, p_values
        
    
def main():
    student = DataScienceStudent(
        name="Michael Frasunkiewicz", 
        age=25, 
        date_of_registration="01.10.2024", 
        registration_number=2559355,
        finished_courses=[
            "Sensortechnik",
            "Signalverarbeitung",
            "Medical Data Science", 
            "Machine Learning",
            "Nanorobotik"
        ],
        favorite_course="Nanorobotik"
    )
    
    #student.plot_integral((-1.5, 12), (-0.78, 1.75), lambda x: np.exp(-x) * np.cos(x))
    #student.plot_integral((-5, 5), (0, 5), lambda x: x ** 2)
    #student.solve_SLE([[1, 2], [3, 4]], [5, 6])
    #student.solve_SLE(
    #    [[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]], 
    #    [4, 1, 3, 2]
    #)
    
    x = np.linspace(0, 10)
    y = x ** 2 + 5
    z = y + 0.1 * np.max(y) * rng.standard_normal(len(y))
    
    X = np.column_stack([np.ones_like(x), x, x ** 2])

    beta, _, _ = student.solve_OLS(z, X)
    
    plt.scatter(x, z)
    plt.plot(x, X @ beta, c="orange")
    plt.show()


if __name__ == "__main__":
    main()
        