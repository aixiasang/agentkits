from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import ReActAgent, Toolkit, ToolResponse, handoff
from _shared import RunPrinter, ali_model, print_result


def build_travel_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def get_weather(city: str, month: str) -> ToolResponse:
        """Get a simple seasonal weather note.

        Args:
            city: Destination city.
            month: Travel month.
        """
        payload = {
            "city": city,
            "month": month,
            "weather": "warm, humid, occasional afternoon showers",
            "packing": "light rain jacket and breathable clothes",
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def search_hotels(city: str, max_price_usd: int) -> ToolResponse:
        """Search hotel options under a nightly budget.

        Args:
            city: Destination city.
            max_price_usd: Maximum nightly room budget in USD.
        """
        payload = [
            {"name": "Riverfront Business Hotel", "nightly_usd": 118},
            {"name": "Metro Hub Suites", "nightly_usd": 96},
            {"name": "Old Town Courtyard", "nightly_usd": 142},
        ]
        filtered = [row for row in payload if row["nightly_usd"] <= max_price_usd]
        return ToolResponse.from_value(json.dumps(filtered))

    @tk.tool()
    def estimate_transport(city: str, travelers: int, days: int) -> ToolResponse:
        """Estimate local transport cost.

        Args:
            city: Destination city.
            travelers: Number of travelers.
            days: Number of travel days.
        """
        total = travelers * days * 18
        return ToolResponse.from_value(
            json.dumps({"city": city, "estimated_usd": total}),
        )

    @tk.tool()
    def build_day_plan(city: str, days: int, style: str) -> ToolResponse:
        """Build a compact itinerary skeleton.

        Args:
            city: Destination city.
            days: Number of travel days.
            style: Preferred travel style.
        """
        plan = [
            f"Day 1: arrive in {city}, check in, riverside dinner",
            f"Day 2: {style} city walk, museum, local food market",
            f"Day 3: half-day workshop block, airport transfer",
        ][:days]
        return ToolResponse.from_value(json.dumps(plan))

    return tk


async def main() -> None:
    user_input = (
        "This is a travel planning request. Please hand it to the travel "
        "specialist: plan a 3-day team offsite in Guangzhou in June for "
        "4 people, hotel budget under 130 USD per night, practical itinerary, "
        "weather note, and local transport estimate."
    )
    printer = RunPrinter("07 handoff demo: router to specialist")
    printer.start(user_input)

    async with ali_model() as model:
        travel_agent = ReActAgent(
            name="travel_agent",
            description=(
                "Plans trips with weather, hotel, itinerary, and transport tools."
            ),
            model=model,
            toolkit=build_travel_toolkit(),
            system_prompt=(
                "You are the travel specialist. Use tools for weather, hotels, "
                "transport, and itinerary. Produce a concise final plan."
            ),
            max_iterations=8,
        )

        router_tk = Toolkit()
        router_tk.register_tool_function(handoff(travel_agent))

        router = ReActAgent(
            name="router",
            model=model,
            toolkit=router_tk,
            system_prompt=(
                "You are a router. For travel planning requests, call "
                "`transfer_to_travel_agent` with a clear reason. Do not solve "
                "travel requests yourself."
            ),
            max_iterations=3,
        )

        result = await router.run(
            user_input,
            on_message=printer.on_message,
        )

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
