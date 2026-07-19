"""OOP, iterator, property ve hata yönetimi kullanan sipariş sistemi."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Product:
    name: str
    category: str
    price: float

class Cart:
    def __init__(self) -> None:
        self._items: list[tuple[Product, int]] = []

    def add(self, product: Product, quantity: int = 1) -> None:
        if quantity < 1:
            raise ValueError("Adet en az 1 olmalı")
        self._items.append((product, quantity))

    def __iter__(self):
        return iter(self._items)

    @property
    def total(self) -> float:
        return round(sum(product.price * quantity for product, quantity in self), 2)

    def category_totals(self) -> dict[str, float]:
        result = {}
        for product, quantity in self:
            result[product.category] = result.get(product.category, 0) + product.price * quantity
        return result

if __name__ == "__main__":
    cart = Cart()
    cart.add(Product("Kulaklık", "Elektronik", 1299), 2)
    cart.add(Product("Defter", "Kırtasiye", 85), 3)
    print("Toplam:", cart.total)
    print("Kategori toplamları:", cart.category_totals())
