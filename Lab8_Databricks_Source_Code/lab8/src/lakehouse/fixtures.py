"""Sample landing files, shaped like what ADF actually lands.

Three batches. Batch 3 is the interesting one: it contains a customer who
changes region TWICE in the same file, which is the case that makes a naive
Delta MERGE throw.
"""

from __future__ import annotations

from pathlib import Path

ORDERS_HEADER = "order_id,customer_id,product_id,quantity,unit_price,order_ts,status,updated_at"

# Batch 1 — a clean day, plus four deliberately broken rows.
ORDERS_BATCH_1 = [
    "ORD-0001,CUST-01,PROD-01,3,1200.50,2024-04-01 10:15:00,placed,2024-04-01 10:15:00",
    "ORD-0002,CUST-02,PROD-02,1,899.00,2024-04-01 11:00:00,shipped,2024-04-01 11:05:00",
    "ORD-0003,CUST-01,PROD-03,2,450.00,2024-04-01 12:30:00,delivered,2024-04-01 18:00:00",
    "ORD-0004,CUST-03,PROD-01,5,1200.50,2024-04-01 13:00:00,placed,2024-04-01 13:00:00",
    "ORD-0005,CUST-02,PROD-04,4,320.75,2024-04-01 14:20:00,shipped,2024-04-01 15:00:00",
    # A duplicate of ORD-0002 with a LATER updated_at. Deduplication must keep
    # this one, and it must do so deterministically.
    "ORD-0002,CUST-02,PROD-02,1,899.00,2024-04-01 11:00:00,delivered,2024-04-01 19:00:00",
    # --- rows that must be quarantined, each breaking exactly one rule -------
    ",CUST-04,PROD-02,1,100.00,2024-04-01 15:00:00,placed,2024-04-01 15:00:00",
    "ORD-0007,CUST-04,PROD-02,-2,100.00,2024-04-01 15:10:00,placed,2024-04-01 15:10:00",
    "ORD-0008,CUST-04,PROD-02,N/A,100.00,2024-04-01 15:20:00,placed,2024-04-01 15:20:00",
    "ORD-0009,CUST-04,PROD-02,1,,2024-04-01 15:30:00,placed,2024-04-01 15:30:00",
]

# Batch 2 — a normal next day.
ORDERS_BATCH_2 = [
    "ORD-0010,CUST-01,PROD-02,2,899.00,2024-04-02 09:00:00,placed,2024-04-02 09:00:00",
    "ORD-0011,CUST-05,PROD-01,1,1200.50,2024-04-02 10:00:00,placed,2024-04-02 10:00:00",
    # A CORRECTION to an order from batch 1. Same key, later updated_at.
    "ORD-0004,CUST-03,PROD-01,5,1200.50,2024-04-01 13:00:00,cancelled,2024-04-02 08:00:00",
]

# Batch 3 — late-arriving data. Placed three weeks ago, arriving now.
ORDERS_BATCH_3 = [
    "ORD-9001,CUST-02,PROD-03,4,450.00,2024-03-12 14:00:00,shipped,2024-04-03 09:00:00",
]

CUSTOMERS_HEADER = "customer_id,full_name,email,city,region,segment,updated_at"

CUSTOMERS_BATCH_1 = [
    "CUST-01,Ravi Kumar,ravi@example.com,Chennai,South,Retail,2024-04-01 08:00:00",
    "CUST-02,Priya Sharma,priya@example.com,Pune,West,SME,2024-04-01 08:00:00",
    "CUST-03,Anil Verma,anil@example.com,Delhi,North,Enterprise,2024-04-01 08:00:00",
    "CUST-04,Sunita Rao,sunita@example.com,Kolkata,East,Retail,2024-04-01 08:00:00",
    "CUST-05,Mohan Das,mohan@example.com,Hyderabad,South,SME,2024-04-01 08:00:00",
]

CUSTOMERS_BATCH_2 = [
    # A TRACKED change: region and segment. Must open a new version.
    "CUST-01,Ravi Kumar,ravi@example.com,Bengaluru,West,Enterprise,2024-04-02 10:00:00",
    # A TYPE 1 change: email only. Must be overwritten in place, no new version.
    "CUST-02,Priya Sharma,priya.sharma@example.com,Pune,West,SME,2024-04-02 10:30:00",
]

# Batch 3 — THE case. CUST-03 changes region TWICE in one file.
# A naive MERGE throws: multiple source rows matched the same target row.
CUSTOMERS_BATCH_3 = [
    "CUST-03,Anil Verma,anil@example.com,Delhi,Central,Enterprise,2024-04-03 09:00:00",
    "CUST-03,Anil Verma,anil@example.com,Mumbai,West,Enterprise,2024-04-03 16:00:00",
]


def write_landing(base: Path, batch: int) -> tuple[Path, Path]:
    """Write one batch's landing files, partitioned by load date like ADF does."""
    load_date = f"2024-04-{batch:02d}"
    orders = {1: ORDERS_BATCH_1, 2: ORDERS_BATCH_2, 3: ORDERS_BATCH_3}[batch]
    customers = {1: CUSTOMERS_BATCH_1, 2: CUSTOMERS_BATCH_2, 3: CUSTOMERS_BATCH_3}[batch]

    orders_dir = base / "orders" / f"load_date={load_date}"
    customers_dir = base / "customers" / f"load_date={load_date}"
    orders_dir.mkdir(parents=True, exist_ok=True)
    customers_dir.mkdir(parents=True, exist_ok=True)

    (orders_dir / "orders.csv").write_text(
        "\n".join([ORDERS_HEADER, *orders]) + "\n", encoding="utf-8")
    (customers_dir / "customers.csv").write_text(
        "\n".join([CUSTOMERS_HEADER, *customers]) + "\n", encoding="utf-8")
    return orders_dir, customers_dir
