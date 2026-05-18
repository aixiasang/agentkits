from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import Toolkit, ToolResponse
from _shared import PlanProgressAgent, RunPrinter, ali_model, print_result


def build_procurement_toolkit() -> Toolkit:
    tk = Toolkit()
    catalog = {
        "developer laptop": {"sku": "LAP-DEV-14", "price": 1299, "stock": 12},
        "4k monitor": {"sku": "MON-4K-27", "price": 329, "stock": 30},
        "usb c dock": {"sku": "DOCK-USBC", "price": 119, "stock": 18},
    }

    @tk.tool()
    def search_inventory(item: str) -> ToolResponse:
        """Look up inventory and unit price for one requested item.

        Args:
            item: Product name or approximate product description.
        """
        key = _best_catalog_key(item, catalog)
        payload = {"requested": item, "matched": key, **catalog[key]}
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def calculate_line_total(unit_price: float, quantity: int) -> ToolResponse:
        """Calculate a line total.

        Args:
            unit_price: Unit price from inventory lookup.
            quantity: Requested item count.
        """
        return ToolResponse.from_value(str(round(unit_price * quantity, 2)))

    @tk.tool()
    def add_costs(a: float, b: float) -> ToolResponse:
        """Add two cost values.

        Args:
            a: First cost value.
            b: Second cost value.
        """
        return ToolResponse.from_value(str(round(a + b, 2)))

    @tk.tool()
    def apply_discount(subtotal: float, discount_code: str) -> ToolResponse:
        """Apply a discount code to a subtotal.

        Args:
            subtotal: Current subtotal before discount.
            discount_code: Discount code supplied by the user.
        """
        rate = 0.10 if discount_code.upper() == "EDU10" else 0.0
        payload = {
            "subtotal": round(subtotal, 2),
            "discount_code": discount_code,
            "discount": round(subtotal * rate, 2),
            "total": round(subtotal * (1 - rate), 2),
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def estimate_delivery(city: str, priority: bool) -> ToolResponse:
        """Estimate delivery window.

        Args:
            city: Destination city.
            priority: Whether priority delivery is requested.
        """
        days = "2-3 business days" if priority else "5-7 business days"
        return ToolResponse.from_value(f"{city}: {days}")

    return tk


def _best_catalog_key(item: str, catalog: dict[str, dict]) -> str:
    text = item.lower()
    for key in catalog:
        if key in text or all(part in text for part in key.split()[:2]):
            return key
    if "monitor" in text:
        return "4k monitor"
    if "dock" in text:
        return "usb c dock"
    return "developer laptop"


async def main() -> None:
    user_input = (
        "Prepare a procurement quote for the Shenzhen engineering room: "
        "3 developer laptops, 5 4k monitors, and 3 usb c docks. Apply "
        "EDU10, use priority delivery, and return a concise purchase summary."
    )
    printer = RunPrinter("06 plan progress: procurement quote")
    printer.start(user_input)

    async with ali_model() as model:
        agent = PlanProgressAgent(
            model=model,
            toolkit=build_procurement_toolkit(),
            max_steps=6,
            max_iterations=10,
            on_plan=printer.plan,
            system_prompt=(
                "Use tools for inventory, arithmetic, discount, and delivery. "
                "Keep the final answer concise and include the final total."
            ),
        )
        result = await agent.run(
            user_input,
            on_message=printer.on_message,
        )

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
