import pytest
from student import TUDStudent, DataScienceStudent


@pytest.fixture
def student():
    return TUDStudent(name="Michael Frasunkiewicz", 
                         age=25, 
                         date_of_registration="01.10.2024", 
                         registration_number=2559355,
                         finished_courses=["Sensortechnik",
                                           "Signalverarbeitung",
                                           "Medical Data Science", 
                                           "Machine Learning",
                                           "Nanorobotik"],
                         favorite_course="Nanorobotik")
    
    
def test_get_student(student):
    # Test getting student's attributes 
    assert student.name == "Michael Frasunkiewicz"
    assert student.age == 25
    assert student.date_of_registration == "01.10.2024"
    assert student.registration_number == 2559355
    assert student.finished_courses == ["Sensortechnik",
                                       "Signalverarbeitung",
                                       "Medical Data Science", 
                                       "Machine Learning",
                                       "Nanorobotik"]
    assert student.favorite_course == "Nanorobotik"
    
    
def test_set_student(student):
    # Test adding a finished course
    student.finished_courses.append("Robust Data Science")
    assert student.finished_courses == ["Sensortechnik",
                                       "Signalverarbeitung", 
                                       "Medical Data Science", 
                                       "Machine Learning",
                                       "Nanorobotik",
                                       "Robust Data Science"]
    
    # Test changing favorite course
    student.favorite_course = "Robust Data Science"
    assert student.favorite_course == "Robust Data Science"
    