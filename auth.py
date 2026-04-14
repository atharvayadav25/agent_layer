# auth.py

USERS_DB = {
    "Sashwat": {"password": "admin123"},
    "Sarthak": {"password": "pass123"},
    "Atharva": {"password": "pass456"},
    "Modussir": {"password": "pass789"}
}

def login_user(username, password):
    if username in USERS_DB and USERS_DB[username]["password"] == password:
        return True
    return False