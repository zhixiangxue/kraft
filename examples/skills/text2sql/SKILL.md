---
name: text2sql-ecommerce
description: Convert natural language questions into PostgreSQL queries for an e-commerce analytics database with schema-aware precision.
---

# Text-to-SQL: E-Commerce Analytics Database

Convert natural language questions into precise PostgreSQL queries for the
analytics database described below. Follow the schema exactly — do NOT
guess table or column names.

## Database Schema

```sql
CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    full_name     VARCHAR(200) NOT NULL,
    segment       VARCHAR(50) CHECK (segment IN ('enterprise','smb','consumer')),
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    sku           VARCHAR(50) UNIQUE NOT NULL,
    name          VARCHAR(300) NOT NULL,
    category      VARCHAR(100) NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    is_active     BOOLEAN DEFAULT TRUE
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    status        VARCHAR(30) CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
    ordered_at    TIMESTAMP WITH TIME ZONE DEFAULT now(),
    shipped_at    TIMESTAMP WITH TIME ZONE,
    total_amount  NUMERIC(12,2) NOT NULL
);

CREATE TABLE order_items (
    item_id       SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(10,2) NOT NULL,
    discount_pct  NUMERIC(5,2) DEFAULT 0
);

CREATE TABLE reviews (
    review_id     SERIAL PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER CHECK (rating BETWEEN 1 AND 5),
    body          TEXT,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

## Query Guidelines

1. **Always qualify ambiguous columns** with table aliases (e.g. `o.customer_id`).
2. **Date filtering**: use `ordered_at >= CURRENT_DATE - INTERVAL '...'` for
   relative ranges; never hard-code dates unless the user specifies them.
3. **Monetary calculations**: use `SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100))` for revenue (not `orders.total_amount`, which excludes item-level discounts).
4. **NULL handling**: `shipped_at IS NULL` means not yet shipped. Cancelled
   orders (`status = 'cancelled'`) should be excluded from revenue/delivery
   metrics unless explicitly asked for.
5. **Pagination**: append `LIMIT 10` by default for "top N" questions when N is
   not specified; always include an explicit `ORDER BY`.
6. **CTEs over subqueries**: prefer `WITH ... AS (...)` for readability when
   more than one aggregation level is involved.
7. **No destructive statements**: never generate INSERT/UPDATE/DELETE.
