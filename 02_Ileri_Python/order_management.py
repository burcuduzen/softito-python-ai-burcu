"""İleri Python konularını kullanan e-ticaret sipariş yönetim sistemi."""
from __future__ import annotations
import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator
from uuid import uuid4

class OrderStatus(str, Enum):
    CREATED = "created"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    unit_price: float
    stock: int

    def __post_init__(self) -> None:
        if not self.sku.strip() or not self.name.strip():
            raise ValueError("SKU ve ürün adı boş olamaz.")
        if self.unit_price <= 0:
            raise ValueError("Fiyat pozitif olmalıdır.")
        if self.stock < 0:
            raise ValueError("Stok negatif olamaz.")

@dataclass
class CartItem:
    product: Product
    quantity: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.quantity <= self.product.stock:
            raise ValueError("Sipariş adedi stok aralığında olmalıdır.")

    @property
    def subtotal(self) -> float:
        return round(self.product.unit_price * self.quantity, 2)

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, subtotal: float) -> float:
        raise NotImplementedError

class NoDiscount(DiscountStrategy):
    def calculate(self, subtotal: float) -> float:
        return 0.0

class PercentageDiscount(DiscountStrategy):
    def __init__(self, rate: float) -> None:
        if not 0 <= rate <= 1:
            raise ValueError("İndirim oranı 0-1 aralığında olmalıdır.")
        self.rate = rate

    def calculate(self, subtotal: float) -> float:
        return round(subtotal * self.rate, 2)

class ThresholdDiscount(DiscountStrategy):
    def __init__(self, threshold: float, amount: float) -> None:
        self.threshold = threshold
        self.amount = amount

    def calculate(self, subtotal: float) -> float:
        return min(self.amount, subtotal) if subtotal >= self.threshold else 0.0

@dataclass
class Customer:
    customer_id: str
    full_name: str
    email: str

    def __post_init__(self) -> None:
        if "@" not in self.email:
            raise ValueError("Geçersiz e-posta adresi.")

class ShoppingCart:
    def __init__(self) -> None:
        self._items: dict[str, CartItem] = {}

    def add(self, product: Product, quantity: int = 1) -> None:
        new_quantity = quantity
        if product.sku in self._items:
            new_quantity += self._items[product.sku].quantity
        self._items[product.sku] = CartItem(product, new_quantity)

    def remove(self, sku: str) -> None:
        if sku not in self._items:
            raise KeyError(f"Sepette {sku} kodlu ürün bulunamadı.")
        del self._items[sku]

    def __iter__(self) -> Iterator[CartItem]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return sum(item.quantity for item in self)

    @property
    def subtotal(self) -> float:
        return round(sum(item.subtotal for item in self), 2)

    def category_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self:
            category = item.product.category
            totals[category] = totals.get(category, 0) + item.subtotal
        return {key: round(value, 2) for key, value in totals.items()}

@dataclass
class Order:
    customer: Customer
    items: list[CartItem]
    discount: float
    order_id: str = field(default_factory=lambda: str(uuid4())[:8].upper())
    status: OrderStatus = OrderStatus.CREATED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def subtotal(self) -> float:
        return round(sum(item.subtotal for item in self.items), 2)

    @property
    def tax(self) -> float:
        return round((self.subtotal - self.discount) * 0.20, 2)

    @property
    def total(self) -> float:
        return round(self.subtotal - self.discount + self.tax, 2)

    def change_status(self, new_status: OrderStatus) -> None:
        valid = {
            OrderStatus.CREATED: {OrderStatus.PAID, OrderStatus.CANCELLED},
            OrderStatus.PAID: {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED: set(),
            OrderStatus.CANCELLED: set(),
        }
        if new_status not in valid[self.status]:
            raise ValueError(f"{self.status} durumundan {new_status} durumuna geçilemez.")
        self.status = new_status

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "customer": asdict(self.customer),
            "items": [
                {
                    "sku": item.product.sku,
                    "name": item.product.name,
                    "quantity": item.quantity,
                    "unit_price": item.product.unit_price,
                    "subtotal": item.subtotal,
                }
                for item in self.items
            ],
            "subtotal": self.subtotal,
            "discount": self.discount,
            "tax": self.tax,
            "total": self.total,
            "status": self.status.value,
            "created_at": self.created_at,
        }

class OrderService:
    def __init__(self, discount_strategy: DiscountStrategy | None = None) -> None:
        self.discount_strategy = discount_strategy or NoDiscount()
        self.orders: list[Order] = []

    def checkout(self, customer: Customer, cart: ShoppingCart) -> Order:
        if len(cart) == 0:
            raise ValueError("Boş sepet için sipariş oluşturulamaz.")
        discount = self.discount_strategy.calculate(cart.subtotal)
        order = Order(customer=customer, items=list(cart), discount=discount)
        self.orders.append(order)
        return order

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([order.to_dict() for order in self.orders], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

def create_demo_order() -> OrderService:
    products = [
        Product("ELK-001", "Kablosuz Kulaklık", "Elektronik", 1299.90, 20),
        Product("KRT-010", "Çizgisiz Defter", "Kırtasiye", 85.00, 100),
        Product("KIT-025", "Python Veri Bilimi", "Kitap", 420.00, 15),
    ]
    cart = ShoppingCart()
    cart.add(products[0], 2)
    cart.add(products[1], 3)
    cart.add(products[2])
    customer = Customer("C-1001", "Burcu Düzen", "burcu@example.com")
    service = OrderService(ThresholdDiscount(threshold=2000, amount=250))
    order = service.checkout(customer, cart)
    order.change_status(OrderStatus.PAID)
    return service

def main() -> None:
    parser = argparse.ArgumentParser(description="E-ticaret sipariş uygulaması")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs" / "orders.json"
    )
    args = parser.parse_args()
    service = create_demo_order()
    service.save(args.output)
    order = service.orders[0]
    print(json.dumps(order.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nSipariş çıktısı: {args.output.resolve()}")

if __name__ == "__main__":
    main()
