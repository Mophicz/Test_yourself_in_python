import numpy as np
import matplotlib.pyplot as plt

from utils import integral, derivative


class TUDStudent:
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
    def solve_integral(self, x_limits, x_stats, f):
        try:
            x_start, x_end = x_limits
        except (TypeError, ValueError): 
            raise TypeError("x_limits must be a Sequence of two numbers.")
        
        try:
            a, b = x_stats
        except (TypeError, ValueError): 
            raise TypeError("x_stats must be a Sequence of two numbers.")    
        
        # 1. Compute y and plot the function
        x = np.linspace(x_start, x_end, int(np.ceil(1000*(x_end - x_start))))
        y = f(x)
        
        # 2. Compute mean, variance, and standard deviation
        # Integrate f(x) over [a, b] using midpoint Riemann sum
        Y_ab, y_ab, x_ab, dx = integral(f, a, b, n=int(np.ceil(10*(b - a))))
        
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
            "extrema_index": extrema_idx
        }
        
    def plot_integral(
        self, x_limits, 
        x_stats, 
        f=lambda x: np.exp(-x) * np.cos(x), 
        derivative=True
    ):
        results = self.solve_integral(x_limits, x_stats, f)
    
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(results["x"], results["y"], label="f(x)")
        ax.bar(
            results["x_interval"], 
            results["y_interval"], 
            width=results["dx"], 
            color="gray", 
            alpha=0.5, 
            ec="gray",
            label=f"Area under f(x) on [{x_stats[0]}, {x_stats[1]}]"
        )
        if derivative:
            plt.plot(results["x"][1:-1], results["dydx"], label="f'(x)")
            plt.scatter(
                results["x"][results["extrema_index"]], 
                results["y"][results["extrema_index"]], 
                color="red",
                label="Extrema"
            )
        plt.legend()
        plt.show()


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
    
    student.plot_integral((-1.5, 12), (0, 1.2), lambda x: np.exp(-x) * np.cos(x))


if __name__ == "__main__":
    main()
        