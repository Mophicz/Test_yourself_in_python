import numpy as np
import matplotlib.pyplot as plt

from utils import integral


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
    def solve_integral(
        self,  
        x_limits, 
        x_stats,
        f=lambda x: np.exp(-x) * np.cos(x),
    ):
        try:
            x_start, x_end = x_limits
        except (TypeError, ValueError): 
            raise TypeError("x_limits must be a Sequence of two numbers.")
        
        try:
            a, b = x_stats
        except (TypeError, ValueError): 
            raise TypeError("x_stats must be a Sequence of two numbers.")    
        
        # 1. Compute y and plot the function
        x = np.linspace(x_start, x_end)
        y = f(x)
        
        # 2. Compute mean, variance, and standard deviation
        mean = integral(f, a, b, 1000) / (b - a)
        var = integral(lambda x: (f(x) - mean) ** 2, a, b, 1000) / (b - a)
        sd = np.sqrt(var)
        
        # 3. Find threshold 70% of y is below
        y_sorted = np.sort(y)
        idx = int(0.7 * len(y_sorted))
        y_m = y_sorted[idx]
        
        
        print(
            f"Mean: {mean:.6f}, "
            f"Variance: {var:.6f}, "
            f"Standard Deviation: {sd:.6f}, "
            f"70th Percentile: {y_m:.6f}"
        )
        
        

        fig, ax = plt.subplots(layout="constrained")
        ax.plot(x, y)
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
    
    student.solve_integral((0, 12), (4, 7))
    

if __name__ == "__main__":
    main()
        