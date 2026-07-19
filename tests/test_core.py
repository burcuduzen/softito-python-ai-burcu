"""Hızlı çalışan temel proje testleri."""
import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[1]

def load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def test_student_average_status_and_validation():
    module = load("01_Python_Temelleri/student_analysis.py", "student_analysis")
    student = module.Student(1, "Test", "Kadın", True, 60, 70, 80)
    assert student.average == 70
    assert student.status == "Başarılı"
    assert student.strongest_course == "writing"
    with pytest.raises(ValueError):
        module.Student(2, "Hatalı", "Erkek", False, 101, 50, 50)

def test_student_summary():
    module = load("01_Python_Temelleri/student_analysis.py", "student_summary")
    students = module.create_demo_students()
    summary = module.create_summary(students)
    assert summary["student_count"] == len(students)
    assert 0 <= summary["success_rate"] <= 1
    assert len(summary["top_three"]) == 3

def test_cart_total_and_discount():
    module = load("02_Ileri_Python/order_management.py", "order_management")
    cart = module.ShoppingCart()
    cart.add(module.Product("T-1", "Ürün", "Test", 25.5, 10), 2)
    assert cart.subtotal == 51
    strategy = module.PercentageDiscount(.10)
    assert strategy.calculate(cart.subtotal) == 5.1

def test_invalid_cart_quantity():
    module = load("02_Ileri_Python/order_management.py", "order_validation")
    product = module.Product("T-2", "Ürün", "Test", 10, 2)
    cart = module.ShoppingCart()
    with pytest.raises(ValueError):
        cart.add(product, 3)

def test_drift_is_small_for_same_distribution():
    module = load("09_Model_Izleme/drift_monitoring.py", "drift_monitoring")
    values = list(range(100))
    assert module.population_stability_index(values, values) < .01

def test_drift_detects_shift():
    module = load("09_Model_Izleme/drift_monitoring.py", "drift_shift")
    reference = list(range(100))
    current = list(range(100, 200))
    assert module.population_stability_index(reference, current) > .25

def test_hotel_demo_schema_and_cleaning():
    module = load("03_EDA/hotel_booking_eda.py", "hotel_booking_eda")
    raw = module.create_demo_data(30)
    clean, quality = module.clean_data(raw)
    assert len(raw) == 40
    assert len(clean) == 30
    assert {"hotel", "adr", "is_canceled", "revenue_potential"}.issubset(clean.columns)
    assert quality["after"]["duplicates"] == 0

def test_rag_chunk_overlap():
    module = load("13_LLM_RAG/turkish_rag.py", "turkish_rag")
    document = module.Document("test.md", " ".join(f"kelime{i}" for i in range(30)))
    chunks = module.split_into_chunks(document, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert chunks[0].end_word == 10
    assert chunks[1].start_word == 8

def test_api_probability_range():
    module = load("14_MLOps_Docker/app.py", "mlops_app")
    customer = module.CustomerFeatures(
        customer_id=" c-1 ", tenure=3, monthly_charges=90,
        total_charges=250, contract="Month-to-month",
        internet_service="Fiber optic", payment_method="Electronic check",
    )
    probability = module.calculate_probability(customer)
    assert 0 <= probability <= 1
    assert customer.customer_id == "C-1"
