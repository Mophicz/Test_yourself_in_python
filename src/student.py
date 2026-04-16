import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


from src.utils import (
    is_number, 
    is_singular, 
    integral, 
    derivative, 
    lu_decomposition,
)


class TUDStudent:
    """Create a student at the Technical University of Darmstadt.
    
    Attributes:
        name (str): The student's name.
        age (int): The student's age.
        date_of_registration (str): The date the student registered at the 
            university.
        registration_number (str): The student's registration number.
        finished_courses (list): List of courses the student has completed.
        favorite_course (str): The student's favorite course.

    Subclasses:
        DataScienceStudent: A student specializing in data science.
    """
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
    """Create a data science student at the Technical University of Darmstadt.

    Inherets from TUDStudent and adds methods for solving math problems.

    Methods:
        solve_integral: Solves an integral and computes statistics.
        plot_integral: Helper method to plot the data from solve_integral.
        solve_SLE: Solves a system of linear equations.
        invert_matrix: Helper method to invert a matrix using solve_SLE.
        solve_MLS: Solves a multivariate least-squares regression problem.
    """
    def solve_integral(
        self, 
        x_limits, 
        x_stats, 
        f=lambda x: np.exp(-x) * np.cos(x),
        n=1000
    ):
        """Solves an integral and calculates statistics.

        Args:
            x_limits (list): Start and end value for domain of f(x).
            x_stats (list): Start and end value for statistics calculation.
            f (Callable, optional): Function f(x) of which statistics are 
                calculated. Defaults to lambda x: np.exp(-x)*np.cos(x).
            n (int, optional): Number of points for numerical integration. 
                Defaults to 1000.

        Raises:
            TypeError: If x_start or x_end are not Sequences (list, tuple) 
            of two numbers.

        Returns:
            dict[str, stats]: A dictionary containing the following keys
                and values:
                - "x": The x values as an array.
                - "y": The corresponding y values of f(x) as an array.
                - "x_interval": The x values shifted by dx/2 for plotting the
                    area under the curve using bars.
                - "y_interval": The corresponding shifted y values.
                - "mean": The mean of f(x) over the interval given in x_stats.
                - "variance": The variance of f(x) over the interval given in
                    x_stats.
                - "standard_deviation": The standard deviation of f(x) over the
                    interval given in x_stats.
                - "threshold_70_percent": The value y_m such that 70% of the
                    y values of f(x) over the interval given in x_stats are 
                    below y_m.
                - "dx": The step size used for numerical integration.
                - "dydx": The numerical derivative of f(x) as an array.
                - "extrema_index": The indices of the extrema of f(x) as an 
                    array.
                - "extrema_x": The x values of the extrema of f(x) as an array.
        """
        # Check input types
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
        
        # 1. Generate x on interval given in x_limits and compute y = f(x)
        x = np.linspace(x_start, x_end, 1000)
        y = f(x)
        
        # 2. Compute mean, variance, and standard deviation
        # Integrate f(x) over [a, b] using midpoint Riemann sum
        Y_ab, y_ab, x_ab, dx = integral(f, a, b, n)
        
        mean = Y_ab / (b - a)
        variance_temp, _, _, _ = integral(lambda x: (f(x) - mean) ** 2, a, b)
        variance = variance_temp / (b - a)
        sd = np.sqrt(variance)
        
        # 3. Find the threshold y_m such that 70% of y is below it
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
        
    def plot_integral(self, results):
        """Helper method to plot the data from solve_integral. 
        
        This allows solve_integral to return the results for testing.
        """
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(results["x"], results["y"], label=r"$y = f(x)$")
        ax.bar(
            results["x_interval"], 
            results["y_interval"], 
            width=results["dx"], 
            color="gray", 
            alpha=0.5, 
            ec="gray",
            label= (
                f"Area under y on [{results["x_interval"][0]:.2g}, "
                f"{results["x_interval"][-1]:.2g}]"
            )
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
        """Solves a system of linear equations Ax = b.

        Args:
            A (array): System matrix A.
            b (array): Solution vector b.

        Raises:
            TypeError: A and b must be convertable to a numpy array of numbers.
            ValueError: If A is singular (since there is no unique solution).

        Returns:
            np.ndarray: Solution vector x.
        """
        # Check input types
        try:
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except Exception:
            raise TypeError(
                "Input must be convertible to a numpy array of numbers."
            )
        
        # Check if there is no unique solution.
        if is_singular(A):
            raise ValueError("Coefficient matrix A is singular.")
        
        # 1. Perform LU decomposition of A
        L, U = lu_decomposition(A)
        
        # 2. Forward substitution to solve Ly = b
        y = np.zeros_like(b, dtype=float)
        
        for i in np.arange(L.shape[0]):
            sum = 0
            for j in np.arange(i):
                sum += y[j] * L[i, j]
                
            y[i] = (b[i] - sum) / L[i, i]
        
        # 3. Backward substitution to solve Ux = y
        x = np.zeros_like(y, dtype=float)
        
        for i in np.arange(U.shape[0] - 1, -1, -1):
            sum = 0
            for j in np.arange(i + 1, U.shape[0]):
                sum += x[j] * U[i, j]
                
            x[i] = (y[i] - sum) / U[i, i]
        
        # 4. Console output
        print(f"Solution to SLE: {x}")
        
        return x
    
    def invert_matrix(self, A):
        """Inverts a matrix by solving SLE column-wise."""
        # Check matrix size, if 1x1 inversion is just 1/A
        if A.size == 1:
            # If A is zero inversion is not possible
            if A == 0:
                raise ValueError("Cannot invert a zero scalar.")
            return 1.0 / A
    
        # Create empty matrix of same dimensio of A
        inv_A = np.zeros(A.shape)

        # Create identity matrix of same size as A
        e = np.eye(len(A))
        
        # Solve column-wise SLE to get the inverse
        for i in np.arange(len(A)):
            inv_A[:, i] = self.solve_SLE(A, e[:, i])
        
        return inv_A
        
    def solve_MLS(self, y, X, output=True):
        """Solves a multivariate least-squares regression problem.

        Args:
            y (array): Response variable.
            X (array): Regressor matrix.
            output (bool, optional): Enables console output of the solution. 
                Defaults to True.

        Raises:
            TypeError: If input is not array_like or does not contain numbers.
            ValueError: If the number of observations is less than the number 
                of variables.

        Returns:
            beta (np.ndarray): The parameter vector beta of the regression 
                model.
            t_stats (np.ndarray): The t-statistics for the parameters.
            p_values (np.ndarray): The p-values for the parameters.
        """
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
            raise ValueError(
                "Number of observations must exceed number of variables."
            )
        
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
          