#!/usr/bin/env python
import json
import secrets
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

LICENSES_PATH = Path("legal/licenses/licenses.json")


def load_licenses():
    if LICENSES_PATH.exists():
        with LICENSES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_licenses(licenses):
    LICENSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LICENSES_PATH.open("w", encoding="utf-8") as f:
        json.dump(licenses, f, indent=2)


def generate_license_key(prefix="CC"):
    # Example key: CC-7F2dc-9e4A1-Ab12T-44C3g
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    blocks = ["".join(secrets.choice(alphabet) for _ in range(5)) for _ in range(5)]
    return f"{prefix}-" + "-".join(blocks)


def next_internal_id(existing):
    if not existing:
        return "LIC-2025-0000001"
    # Assume IDs are like LIC-YYYY-NNNNNNN
    last = sorted(existing, key=lambda x: x["id"])[-1]["id"]
    prefix, year, num = last.split("-")
    num_int = int(num) + 1
    return f"{prefix}-{year}-{num_int:05d}"


def create_license(
    customer_name: str,
    customer_email: str,
    plan: str = "pro",
    seats: int = 1,
    validity_days: int = 365,
    notes: str | None = None,
):
    licenses = load_licenses()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=validity_days)

    license_id = next_internal_id(licenses)
    license_key = generate_license_key()

    record = {
        "id": license_id,
        "key": license_key,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "plan": plan,
        "status": "active",
        "seats": seats,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "notes": notes or "",
    }

    licenses.append(record)
    save_licenses(licenses)

    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a commercial license key.")
    parser.add_argument("--name", required=True, help="Customer name")
    parser.add_argument("--email", required=True, help="Customer email")
    parser.add_argument("--plan", default="pro", help="Plan type (e.g., pro, enterprise)")
    parser.add_argument("--seats", type=int, default=1, help="Number of seats")
    parser.add_argument("--days", type=int, default=365, help="License validity in days")
    parser.add_argument("--notes", default="", help="Internal notes")

    args = parser.parse_args()

    rec = create_license(
        customer_name=args.name,
        customer_email=args.email,
        plan=args.plan,
        seats=args.seats,
        validity_days=args.days,
        notes=args.notes,
    )

    print("✅ License created:")
    print(f"  Internal ID: {rec['id']}")
    print(f"  License Key: {rec['key']}")
    print(f"  Customer:    {rec['customer_name']} <{rec['customer_email']}>")
    print(f"  Plan:        {rec['plan']}  | Seats: {rec['seats']}")
    print(f"  Expires At:  {rec['expires_at']}")
