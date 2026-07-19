"""Öğrenci notlarıyla temel Python, CSV ve JSON uygulaması."""
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

@dataclass
class Student:
    name: str
    math: int
    reading: int
    writing: int

    @property
    def average(self) -> float:
        return round((self.math + self.reading + self.writing) / 3, 2)

    @property
    def status(self) -> str:
        return "Başarılı" if self.average >= 60 else "Geliştirilmeli"

def summarize(students: list[Student]) -> dict:
    return {
        "student_count": len(students),
        "class_average": round(sum(s.average for s in students) / len(students), 2),
        "top_student": max(students, key=lambda s: s.average).name,
    }

if __name__ == "__main__":
    students = [
        Student("Ayşe", 78, 84, 80),
        Student("Can", 55, 62, 58),
        Student("Ece", 92, 88, 95),
    ]
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    rows = [{**asdict(s), "average": s.average, "status": s.status} for s in students]
    with (output / "students.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summarize(students), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summarize(students))
