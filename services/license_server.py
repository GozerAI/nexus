from datetime import datetime, timezone
from pathlib import Path
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LICENSES_PATH = Path("legal/licenses/licenses.json")

app = FastAPI(title="License Verification API")


class LicenseRecord(BaseModel):
    id: str
    key: str
    customer_name: str
    customer_email: str
    plan: str
    status: str
    seats: int
    expires_at: str
    created_at: str
    updated_at: str
    notes: str | None = ""


def load_licenses():
    if LICENSES_PATH.exists():
        with LICENSES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def find_license_by_key(key: str) -> dict | None:
    licenses = load_licenses()
    for lic in licenses:
        if lic["key"] == key:
            return lic
    return None


@app.get("/verify")
def verify_license(key: str):
    """Verify a license key and return basic status information."""
    rec = find_license_by_key(key)
    if not rec:
        raise HTTPException(status_code=404, detail="License not found")

    # Parse expiry
    try:
        expires = datetime.fromisoformat(rec["expires_at"].replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid license record")

    now = datetime.now(timezone.utc)

    status = rec["status"]
    if status != "active":
        return {
            "valid": False,
            "reason": f"License status is {status}",
            "license_id": rec["id"],
            "plan": rec["plan"],
        }

    if now > expires:
        return {
            "valid": False,
            "reason": "License expired",
            "license_id": rec["id"],
            "plan": rec["plan"],
        }

    return {
        "valid": True,
        "reason": "OK",
        "license_id": rec["id"],
        "plan": rec["plan"],
        "seats": rec["seats"],
        "customer_name": rec["customer_name"],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("LICENSE_SERVER_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
