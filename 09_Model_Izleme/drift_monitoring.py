"""Referans ve canlı özellik dağılımları için PSI tabanlı drift alarmı."""
import numpy as np

def population_stability_index(reference, current, bins=10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    reference_ratio = np.histogram(reference, edges)[0] / len(reference)
    current_ratio = np.histogram(current, edges)[0] / len(current)
    reference_ratio = np.clip(reference_ratio, 1e-6, None)
    current_ratio = np.clip(current_ratio, 1e-6, None)
    return float(np.sum(
        (current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio)
    ))

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    reference = rng.normal(20, 5, 5000)
    live = rng.normal(23, 6, 2500)
    psi = population_stability_index(reference, live)
    status = "Drift var" if psi >= .25 else "İzle" if psi >= .10 else "Stabil"
    print({"PSI": round(psi, 4), "status": status})
