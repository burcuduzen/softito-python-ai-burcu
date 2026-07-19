"""Hızlı çalışan temel proje testleri."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]

def load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_student_average_and_status():
    module = load("01_Python_Temelleri/student_analysis.py", "student_analysis")
    student = module.Student("Test", 60, 70, 80)
    assert student.average == 70
    assert student.status == "Başarılı"

def test_cart_total():
    module = load("02_Ileri_Python/order_management.py", "order_management")
    cart = module.Cart()
    cart.add(module.Product("Ürün", "Test", 25.5), 2)
    assert cart.total == 51

def test_drift_is_small_for_same_distribution():
    module = load("09_Model_Izleme/drift_monitoring.py", "drift_monitoring")
    values = list(range(100))
    assert module.population_stability_index(values, values) < 0.01

def test_hotel_demo_schema():
    module = load("03_EDA/hotel_booking_eda.py", "hotel_booking_eda")
    df = module.create_demo_data(20)
    assert len(df) == 20
    assert {"hotel", "adr", "is_canceled"}.issubset(df.columns)
