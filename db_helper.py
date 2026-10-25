import os
import re
import certifi
import bcrypt
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Attempt to load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEFAULT_LOCAL_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "plantcare_db"

_client = None
_active_uri_type = "unknown"


def get_mongo_uri():
    """
    Retrieve MongoDB connection string.
    Priority:
    1. Streamlit secrets (.streamlit/secrets.toml)
    2. Environment variable MONGODB_URI or MONGO_URI (.env or system)
    3. Local MongoDB fallback
    """
    # 1. Check Streamlit secrets
    try:
        import streamlit as st
        for key in ["MONGODB_URI", "MONGO_URI"]:
            if key in st.secrets and st.secrets[key]:
                return st.secrets[key].strip()
    except Exception:
        pass

    # 2. Check environment variable
    for key in ["MONGODB_URI", "MONGO_URI"]:
        env_uri = os.environ.get(key)
        if env_uri and env_uri.strip():
            return env_uri.strip()

    # 3. Fallback to local
    return DEFAULT_LOCAL_URI


def get_db():
    """Connect to MongoDB Atlas (or local fallback) and return database instance."""
    global _client, _active_uri_type

    if _client is not None:
        return _client[DB_NAME]

    uri = get_mongo_uri()
    client_kwargs = {
        "serverSelectionTimeoutMS": 5000,
    }

    # Enable TLS certificate validation with certifi for Atlas SRV connections
    if "mongodb+srv://" in uri or "ssl=true" in uri.lower():
        client_kwargs["tlsCAFile"] = certifi.where()
        _active_uri_type = "Atlas Cluster"
    else:
        _active_uri_type = "Local MongoDB"

    try:
        _client = MongoClient(uri, **client_kwargs)
        # Verify connection with a lightweight ping
        _client.admin.command("ping")
        db = _client[DB_NAME]
        db.users.create_index("email", unique=True)
        db.history.create_index([("user_email", 1), ("timestamp", -1)])
        return db
    except Exception as e:
        print(f"Primary MongoDB connection to {uri} failed: {e}")
        # If Atlas failed or not configured, attempt fallback to local if different
        if uri != DEFAULT_LOCAL_URI:
            try:
                print("Falling back to local MongoDB at 127.0.0.1:27017...")
                _client = MongoClient(DEFAULT_LOCAL_URI, serverSelectionTimeoutMS=2500)
                _client.admin.command("ping")
                _active_uri_type = "Local MongoDB (Atlas Fallback)"
                db = _client[DB_NAME]
                db.users.create_index("email", unique=True)
                db.history.create_index([("user_email", 1), ("timestamp", -1)])
                return db
            except Exception as local_err:
                _client = None
                raise ConnectionError(
                    f"Unable to connect to MongoDB Atlas ({e}) or Local MongoDB ({local_err}). "
                    "Please check your MONGO_URI in .env or .streamlit/secrets.toml."
                )
        else:
            _client = None
            raise e


def get_connection_status():
    """Return description of active connection (Atlas or Local)."""
    try:
        get_db()
        return _active_uri_type, True
    except Exception as e:
        return str(e), False


def validate_email_format(email: str):
    """Validate email syntax strictly."""
    if not email or not isinstance(email, str):
        return False, "Email address cannot be empty."
    email = email.strip()
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g. alex@example.com)."
    if len(email) > 254:
        return False, "Email address is too long."
    return True, None


def validate_password_strength(password: str):
    """Validate password rules."""
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, None


def register_user(name: str, email: str, password: str):
    """Register a new user in MongoDB with bcrypt hashed password."""
    name = (name or "").strip()
    email = (email or "").strip().lower()

    if not name:
        return False, "Please enter your name."

    valid_email, email_err = validate_email_format(email)
    if not valid_email:
        return False, email_err

    valid_pw, pw_err = validate_password_strength(password)
    if not valid_pw:
        return False, pw_err

    try:
        db = get_db()
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user_doc = {
            "name": name,
            "email": email,
            "password_hash": pw_hash,
            "created_at": datetime.now()
        }
        db.users.insert_one(user_doc)
        return True, {"name": name, "email": email}
    except DuplicateKeyError:
        return False, "An account with this email already exists. Please log in."
    except Exception as e:
        return False, f"Database error during registration: {str(e)}"


def authenticate_user(email: str, password: str):
    """Verify credentials and return user info."""
    email = (email or "").strip().lower()
    if not email or not password:
        return False, "Please provide both email and password."

    try:
        db = get_db()
        user = db.users.find_one({"email": email})
        if not user:
            return False, "No account found with this email. Please register."

        stored_hash = user.get("password_hash", "")
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
            return True, {
                "name": user.get("name", "Gardener"),
                "email": user.get("email")
            }
        else:
            return False, "Incorrect password. Please try again."
    except Exception as e:
        return False, f"Database error during login: {str(e)}"


def save_consultation_history(user_email: str, user_name: str, consultation_data: dict):
    """Save an interaction/prediction into MongoDB history collection."""
    if not user_email or not consultation_data:
        return False
    try:
        db = get_db()
        doc = {
            "user_email": user_email.strip().lower(),
            "user_name": user_name,
            "timestamp": datetime.now(),
            **consultation_data
        }
        db.history.insert_one(doc)
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False


def get_user_history(user_email: str, limit: int = 50):
    """Retrieve previous consultations for a user, newest first."""
    if not user_email:
        return []
    try:
        db = get_db()
        cursor = db.history.find(
            {"user_email": user_email.strip().lower()}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []


def clear_user_history(user_email: str):
    """Clear consultation history for a user."""
    if not user_email:
        return False
    try:
        db = get_db()
        db.history.delete_many({"user_email": user_email.strip().lower()})
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False