"""Öğrenci başarı verisiyle kapsamlı Python temelleri uygulaması.

Konular:
- dataclass ve type hint
- liste/sözlük işlemleri
- CSV ve JSON okuma-yazma
- veri doğrulama ve hata yönetimi
- fonksiyonlar, property, sıralama ve gruplama
- komut satırı argümanları
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

PASS_LIMIT = 60
COURSES = ("math", "reading", "writing")

@dataclass
class Student:
    student_id: int
    name: str
    gender: str
    preparation_course: bool
    math: int
    reading: int
    writing: int

    def __post_init__(self) -> None:
        if self.student_id < 1:
            raise ValueError("Öğrenci numarası pozitif olmalıdır.")
        if not self.name.strip():
            raise ValueError("Öğrenci adı boş bırakılamaz.")
        for course in COURSES:
            score = getattr(self, course)
            if not 0 <= score <= 100:
                raise ValueError(f"{course} notu 0-100 arasında olmalıdır.")

    @property
    def average(self) -> float:
        return round(mean([self.math, self.reading, self.writing]), 2)

    @property
    def status(self) -> str:
        if self.average >= 85:
            return "Çok Başarılı"
        if self.average >= PASS_LIMIT:
            return "Başarılı"
        return "Geliştirilmeli"

    @property
    def strongest_course(self) -> str:
        return max(COURSES, key=lambda course: getattr(self, course))

    def to_record(self) -> dict:
        record = asdict(self)
        record.update(
            average=self.average,
            status=self.status,
            strongest_course=self.strongest_course,
        )
        return record

def create_demo_students() -> list[Student]:
    return [
        Student(1, "Ayşe Yılmaz", "Kadın", True, 78, 84, 80),
        Student(2, "Can Demir", "Erkek", False, 55, 62, 58),
        Student(3, "Ece Kaya", "Kadın", True, 92, 88, 95),
        Student(4, "Mert Aydın", "Erkek", True, 71, 69, 74),
        Student(5, "Duru Çelik", "Kadın", False, 48, 57, 51),
        Student(6, "Bora Arslan", "Erkek", True, 87, 81, 79),
        Student(7, "Selin Koç", "Kadın", True, 65, 76, 83),
        Student(8, "Emir Şahin", "Erkek", False, 59, 54, 61),
    ]

def load_students(path: Path) -> list[Student]:
    students = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            try:
                students.append(Student(
                    student_id=int(row["student_id"]),
                    name=row["name"],
                    gender=row["gender"],
                    preparation_course=row["preparation_course"].lower() in {"1", "true", "yes", "evet"},
                    math=int(row["math"]),
                    reading=int(row["reading"]),
                    writing=int(row["writing"]),
                ))
            except (KeyError, ValueError) as exc:
                print(f"Uyarı: {row_number}. satır atlandı: {exc}")
    if not students:
        raise ValueError("Dosyada geçerli öğrenci kaydı bulunamadı.")
    return students

def course_statistics(students: list[Student]) -> dict:
    result = {}
    for course in COURSES:
        scores = [getattr(student, course) for student in students]
        result[course] = {
            "average": round(mean(scores), 2),
            "median": round(median(scores), 2),
            "minimum": min(scores),
            "maximum": max(scores),
        }
    return result

def group_statistics(students: list[Student], attribute: str) -> dict:
    groups: dict[str, list[float]] = {}
    for student in students:
        key = str(getattr(student, attribute))
        groups.setdefault(key, []).append(student.average)
    return {
        key: {"count": len(values), "average": round(mean(values), 2)}
        for key, values in groups.items()
    }

def create_summary(students: list[Student]) -> dict:
    ranked = sorted(students, key=lambda student: student.average, reverse=True)
    successful = [student for student in students if student.average >= PASS_LIMIT]
    return {
        "student_count": len(students),
        "class_average": round(mean(student.average for student in students), 2),
        "success_rate": round(len(successful) / len(students), 4),
        "top_three": [
            {"name": student.name, "average": student.average}
            for student in ranked[:3]
        ],
        "course_statistics": course_statistics(students),
        "gender_statistics": group_statistics(students, "gender"),
        "preparation_statistics": group_statistics(students, "preparation_course"),
    }

def save_outputs(students: list[Student], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [student.to_record() for student in students]
    with (output_dir / "student_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(create_summary(students), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def print_report(students: list[Student]) -> None:
    print("\nÖĞRENCİ BAŞARI RAPORU")
    print("=" * 60)
    for rank, student in enumerate(
        sorted(students, key=lambda item: item.average, reverse=True), start=1
    ):
        print(
            f"{rank:>2}. {student.name:<20} Ortalama: {student.average:>6.2f} "
            f"Durum: {student.status}"
        )
    print("-" * 60)
    summary = create_summary(students)
    print(f"Sınıf ortalaması: %{summary['class_average']}")
    print(f"Başarı oranı: %{summary['success_rate'] * 100:.1f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Öğrenci başarı analizi")
    parser.add_argument("--input", type=Path, help="İsteğe bağlı öğrenci CSV dosyası")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    students = load_students(args.input) if args.input else create_demo_students()
    print_report(students)
    save_outputs(students, args.output)
    print(f"\nÇıktılar kaydedildi: {args.output.resolve()}")

if __name__ == "__main__":
    main()
