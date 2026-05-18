from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import ReActAgent, Toolkit, ToolResponse, handoff
from _shared import PlanProgressAgent, RunPrinter, ali_model, print_result


USER_INPUT = (
    "Customer ACME Robotics says order ORD-9821 is late and blocks Monday's "
    "pilot launch in Shanghai. Build a recovery plan: verify the account, "
    "check order status, find a replacement path, estimate delivery, create "
    "an action ticket, and tell the account team what to say."
)


def build_ops_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def lookup_customer(account: str) -> ToolResponse:
        """Look up customer tier, owner, and escalation policy.

        Args:
            account: Customer or account name.
        """
        payload = {
            "account": account,
            "tier": "enterprise",
            "owner": "Mina Chen",
            "support_sla_hours": 4,
            "escalation_policy": "same-day executive update for launch blockers",
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def check_order(order_id: str) -> ToolResponse:
        """Check fulfillment status for an order.

        Args:
            order_id: Order identifier from the customer request.
        """
        payload = {
            "order_id": order_id,
            "sku": "EDGE-KIT-PRO",
            "quantity": 2,
            "order_value_usd": 8400,
            "status": "carrier exception",
            "delay_days": 3,
            "current_eta": "2026-05-22",
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def check_inventory(sku: str, region: str) -> ToolResponse:
        """Find replacement inventory by SKU and region.

        Args:
            sku: Product SKU.
            region: Fulfillment region.
        """
        payload = {
            "sku": sku,
            "region": region,
            "available": 3,
            "warehouse": "Shanghai Pudong spare pool",
            "release_condition": "manager approval required",
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def estimate_delivery(city: str, priority: bool) -> ToolResponse:
        """Estimate delivery for replacement shipment.

        Args:
            city: Destination city.
            priority: Whether priority courier is used.
        """
        payload = {
            "city": city,
            "priority": priority,
            "eta": "next business day before 12:00",
            "cost_usd": 180,
        }
        return ToolResponse.from_value(json.dumps(payload))

    @tk.tool()
    def calculate_service_credit(order_value_usd: float, delay_days: int) -> ToolResponse:
        """Calculate a simple delay credit.

        Args:
            order_value_usd: Order value in USD.
            delay_days: Number of delay days.
        """
        credit = round(min(order_value_usd * 0.02 * delay_days, 750), 2)
        return ToolResponse.from_value(
            json.dumps({"credit_usd": credit, "cap_usd": 750}),
        )

    @tk.tool()
    def create_action_ticket(
        account: str,
        order_id: str,
        severity: str,
        summary: str,
    ) -> ToolResponse:
        """Create an internal action ticket.

        Args:
            account: Customer account.
            order_id: Related order identifier.
            severity: Incident severity.
            summary: Short operational summary.
        """
        payload = {
            "ticket_id": "OPS-20260517-118",
            "account": account,
            "order_id": order_id,
            "severity": severity,
            "summary": summary,
            "owner": "ops-oncall-apac",
        }
        return ToolResponse.from_value(json.dumps(payload))

    return tk


async def main() -> None:
    printer = RunPrinter("08 reproducible demo: enterprise order recovery")
    printer.start(USER_INPUT)

    async with ali_model() as model:
        ops_agent = PlanProgressAgent(
            name="ops_agent",
            description=(
                "Resolves enterprise order incidents using customer, order, "
                "inventory, logistics, credit, and ticketing tools."
            ),
            model=model,
            toolkit=build_ops_toolkit(),
            max_steps=7,
            max_iterations=12,
            on_plan=printer.plan,
            system_prompt=(
                "You are the operations specialist. Use tools for every factual "
                "lookup or calculation. Keep plan progress visible through tool "
                "calls, then produce a concise recovery plan with owner, ETA, "
                "ticket id, customer message, and residual risk."
            ),
        )

        router_tk = Toolkit()
        router_tk.register_tool_function(handoff(ops_agent))

        router = ReActAgent(
            name="router",
            model=model,
            toolkit=router_tk,
            max_iterations=3,
            system_prompt=(
                "You are an intake router. For enterprise account, order, "
                "fulfillment, launch-blocker, or incident requests, call "
                "`transfer_to_ops_agent` with a clear reason. Do not solve "
                "those requests yourself."
            ),
        )

        result = await router.run(
            USER_INPUT,
            on_message=printer.on_message,
        )

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
