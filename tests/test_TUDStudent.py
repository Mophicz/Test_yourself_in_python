import pytest
    
    
def test_get_student(student):
    assert student.name == "Michael Frasunkiewicz"
    assert student.age == 25
    assert student.date_of_registration == "01.10.2024"
    assert student.registration_number == 2559355
    assert student.finished_courses == [
        "Sensortechnik",
        "Signalverarbeitung",
        "Medical Data Science", 
        "Machine Learning",
        "Nanorobotik"
    ]
    assert student.favorite_course == "Nanorobotik"
    
    
def test_add_course(student):
    courses = student.finished_courses
    courses.append("Robust Data Science")
    assert student.finished_courses == courses


def test_change_favorite_course(student):
    student.favorite_course = "Robust Data Science"
    assert student.favorite_course == "Robust Data Science"
 
 
# Define invalid inputs for finished_courses
@pytest.mark.parametrize(
    ("input, exception"), 
    [
        (1, TypeError),
        ("str", TypeError),
        (["str", 1, "str"], TypeError)
    ],
    ids=[
        "int and not list", 
        "str but not list", 
        "list but integer in list"
        ]
)
def test_set_invalid_finished_courses(student, input, exception):
    """Test setting finished_courses with invalid input."""
    with pytest.raises(exception):
        student.finished_courses = input
        

def test_set_invalid_favorite_course(student):
    """Test setting favorite_course with invalid input."""
    with pytest.raises(TypeError):
        student.favorite_course = 1
        