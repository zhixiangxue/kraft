"""Evaluate a Text-to-SQL skill downloaded from marketplace.

Demonstrates ``kraft.evaluate()`` — the quick-assessment path for skills
you didn't author yourself. This example evaluates a schema-aware
Text-to-SQL skill against natural-language queries.

The skill lives in ``skills/text2sql/`` (simulating a marketplace download).
Without the skill, the baseline model must guess table/column names and
join conditions; with it, the model has the exact schema + query
guidelines pinned, producing consistently correct SQL.

Three ways to supply evaluation cases:
  1. Natural-language strings (active below) — kraft structures them via LLM
  2. Pre-structured EvalCase objects
  3. None — auto-generate from SKILL.md content
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import EvalCase, evaluate

from _report import print_eval_report

# Simulates a skill downloaded from marketplace.
SKILL_DIR = Path(__file__).parent / "skills" / "text2sql"


# Schema shared by all cases — both baseline and with-skill see this.
# The skill's value comes from its GUIDELINES (revenue formula, NULL handling,
# CTE preference, pagination defaults), not from schema awareness alone.
SCHEMA = """\
Given the following PostgreSQL schema:

  customers(customer_id PK, email, full_name, segment ['enterprise','smb','consumer'], created_at)
  products(product_id PK, sku, name, category, unit_price NUMERIC(10,2), is_active BOOL)
  orders(order_id PK, customer_id FK, status ['pending','confirmed','shipped','delivered','cancelled'], ordered_at, shipped_at, total_amount NUMERIC(12,2))
  order_items(item_id PK, order_id FK, product_id FK, quantity INT, unit_price NUMERIC(10,2), discount_pct NUMERIC(5,2) DEFAULT 0)
  reviews(review_id PK, product_id FK, customer_id FK, rating 1-5, body TEXT, created_at)

Write a PostgreSQL query for the following question:"""


async def main():
    api_key = os.environ["OPENAI_API_KEY"]

    # Pre-structured cases with schema embedded in input.
    # Both arms see the schema; the skill's lift comes purely from its
    # query guidelines (revenue formula, exclude cancelled, CTE style, etc.).
    skill = await evaluate(
        skill_dir=SKILL_DIR,
        model="openai/gpt-4o",
        api_key=api_key,
        cases=[
            EvalCase(
                input=f"{SCHEMA}\nTop 5 customers by total revenue in the last 30 days.",
                reference=(
                    "WITH customer_revenue AS (\n"
                    "  SELECT c.customer_id, c.full_name,\n"
                    "    SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue\n"
                    "  FROM customers c\n"
                    "  JOIN orders o ON o.customer_id = c.customer_id\n"
                    "  JOIN order_items oi ON oi.order_id = o.order_id\n"
                    "  WHERE o.status != 'cancelled'\n"
                    "    AND o.ordered_at >= CURRENT_DATE - INTERVAL '30 days'\n"
                    "  GROUP BY c.customer_id, c.full_name\n"
                    ")\n"
                    "SELECT * FROM customer_revenue ORDER BY revenue DESC LIMIT 5;"
                ),
                criterion=(
                    "Must compute revenue from order_items (qty*price*(1-discount/100)), "
                    "not orders.total_amount. Must exclude cancelled orders. "
                    "Must use relative date filter, not hard-coded dates."
                ),
                source="user",
            ),
            EvalCase(
                input=f"{SCHEMA}\nAverage rating per product category.",
                reference=(
                    "SELECT p.category, AVG(r.rating) AS avg_rating\n"
                    "FROM products p\n"
                    "JOIN reviews r ON r.product_id = p.product_id\n"
                    "GROUP BY p.category\n"
                    "ORDER BY avg_rating DESC;"
                ),
                criterion="Must JOIN products and reviews on product_id, GROUP BY category, ORDER BY avg_rating DESC.",
                source="user",
            ),
            EvalCase(
                input=f"{SCHEMA}\nAll orders confirmed but not yet shipped.",
                reference=(
                    "SELECT * FROM orders\n"
                    "WHERE status = 'confirmed' AND shipped_at IS NULL;"
                ),
                criterion="Must filter status='confirmed' AND shipped_at IS NULL. Must not use shipped_at = '' or similar.",
                source="user",
            ),
            EvalCase(
                input=f"{SCHEMA}\nEnterprise customers who have never placed an order.",
                reference=(
                    "SELECT c.*\n"
                    "FROM customers c\n"
                    "LEFT JOIN orders o ON o.customer_id = c.customer_id\n"
                    "WHERE c.segment = 'enterprise' AND o.order_id IS NULL;"
                ),
                criterion="Must use LEFT JOIN + IS NULL (or NOT EXISTS). Must filter segment='enterprise'.",
                source="user",
            ),
            EvalCase(
                input=f"{SCHEMA}\nMonthly revenue trend for the last 12 months, including months with zero revenue.",
                reference=(
                    "WITH months AS (\n"
                    "  SELECT generate_series(\n"
                    "    date_trunc('month', CURRENT_DATE) - INTERVAL '11 months',\n"
                    "    date_trunc('month', CURRENT_DATE),\n"
                    "    INTERVAL '1 month'\n"
                    "  ) AS month_start\n"
                    ")\n"
                    "SELECT m.month_start,\n"
                    "  COALESCE(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)), 0) AS revenue\n"
                    "FROM months m\n"
                    "LEFT JOIN orders o ON date_trunc('month', o.ordered_at) = m.month_start\n"
                    "  AND o.status != 'cancelled'\n"
                    "LEFT JOIN order_items oi ON oi.order_id = o.order_id\n"
                    "GROUP BY m.month_start\n"
                    "ORDER BY m.month_start;"
                ),
                criterion=(
                    "Must use generate_series for 12 month boundaries. Must LEFT JOIN so zero-revenue months appear. "
                    "Must exclude cancelled orders. Must use item-level revenue formula."
                ),
                source="user",
            ),
        ],
    )

    print_eval_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
