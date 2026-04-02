import pytest
from student import TUDStudent, DataScienceStudent


@pytest.fixture
def student():
    return TUDStudent(
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


@pytest.fixture
def data_science_student(student):
    return DataScienceStudent(
        name=student.name,
        age=student.age,
        date_of_registration=student.date_of_registration,
        registration_number=student.registration_number,
        finished_courses=student.finished_courses,
        favorite_course=student.favorite_course
    )    
    