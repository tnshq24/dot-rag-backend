import os
import datetime
from typing import Optional, Tuple

import jwt


def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        # Fallback for development
        secret = "change-this-in-production"
    return secret


def get_jwt_issuer() -> str:
    return os.environ.get("JWT_ISSUER", "dotrag-backend")


def get_jwt_audience() -> str:
    return os.environ.get("JWT_AUDIENCE", "dotrag-frontend")


def generate_access_token(
    subject: str,
    email: str,
    is_admin: bool = False,
    expires_in_minutes: int = 60,
) -> str:
    now = datetime.datetime.utcnow()
    payload = {
        "sub": subject,
        "email": email,
        "admin": is_admin,
        "iss": get_jwt_issuer(),
        "aud": get_jwt_audience(),
        "iat": now,
        "nbf": now,
        "exp": now + datetime.timedelta(minutes=expires_in_minutes),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")


def verify_access_token(token: str) -> Tuple[bool, Optional[dict]]:
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=["HS256"],
            audience=get_jwt_audience(),
            issuer=get_jwt_issuer(),
            options={"require": ["exp", "iat", "nbf"]},
        )
        return True, payload
    except Exception:
        return False, None


