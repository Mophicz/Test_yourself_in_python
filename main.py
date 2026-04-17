import numpy as np
from matplotlib import pyplot as plt


from  src.student import DataScienceStudent


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
    
    
def call_plot_integral():
    results = student.solve_integral(
        x_limits=(-1.5, 12), 
        x_stats=(-0.78, 1.75), 
        f=lambda x: np.exp(-x) * np.cos(x),
        n=10
    )

    student.plot_integral(results)


def call_solve_SLE():
    student.solve_SLE(
        [[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]], 
        [4, 1, 3, 2]
    )
    

def call_solve_MLS():
    x = np.linspace(-10, 10)
    y = x ** 3
    z = y + 0.1 * np.max(y) * np.random.randn(len(y)) + 50
    
    X = np.column_stack([np.ones_like(x), x, x **2, x ** 3])

    beta, _, _ = student.solve_MLS(z, X)
    
    plt.scatter(x, z)
    plt.plot(x, X @ beta, c="orange")
    plt.show()


if __name__ == "__main__":
    call_solve_MLS()
    