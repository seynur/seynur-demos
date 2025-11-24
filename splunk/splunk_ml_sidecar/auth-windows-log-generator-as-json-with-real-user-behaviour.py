#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from datetime import datetime, timedelta

OUT_FILE = "<full-output-file-path>"
TOTAL_EVENTS = 30000

NORMAL_USERS = ["alice", "bob", "charlie", "diana"]
SERVICE_ACCOUNTS = ["svc-app01", "svc-app02", "svc-db", "svc-web", "svc-backup"]
ADMINS = ["administrator", "helpdesk", "adm-john"]
ATTACKER_USERS = ["attacker1", "attacker2"]

SIGNATURES = {
    4624: "An account was successfully logged on",
    4625: "An account failed to log on",
    4634: "An account was logged off",
    4768: "A Kerberos authentication ticket (TGT) was requested",
    4769: "A Kerberos service ticket was requested",
    4672: "Special privileges assigned to new logon"
}

WORKSTATIONS = [f"WORKSTATION{str(i).zfill(3)}" for i in range(1, 200)]
SERVERS = [f"SERVER{str(i).zfill(3)}" for i in range(1, 50)]
INTERNAL_NET = ["10.10.1.", "10.10.2.", "10.15."]

def random_private_ip():
    base = random.choice(INTERNAL_NET)
    return base + str(random.randint(1, 254))

def random_timestamp(days_back=90):
    now = datetime.utcnow()
    delta = timedelta(days=random.randint(0, days_back), seconds=random.randint(0,86400))
    t = now - delta
    return t.isoformat() + "Z"

def generate_normal_user():
    user = random.choice(NORMAL_USERS)
    ts = random_timestamp()
    hour = datetime.fromisoformat(ts.replace("Z","+00:00")).hour

    signature = random.choice([4624, 4634, 4768])
    action = "success" if signature != 4625 else "failure"

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": random_private_ip(),
        "dest": random.choice(WORKSTATIONS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": action,
        "process": "C:\\Windows\\System32\\wininit.exe"
    }

def generate_service_account():
    user = random.choice(SERVICE_ACCOUNTS)
    ts = random_timestamp()
    signature = random.choice([4768, 4769, 4624])

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": random_private_ip(),
        "dest": random.choice(SERVERS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": "success",
        "process": "C:\\Windows\\System32\\lsass.exe"
    }

def generate_admin_user():
    user = random.choice(ADMINS)
    ts = random_timestamp()
    signature = random.choice([4624, 4625, 4672])

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": random_private_ip(),
        "dest": random.choice(WORKSTATIONS + SERVERS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": "success" if signature == 4624 else "failure",
        "process": "C:\\Windows\\System32\\mmc.exe"
    }

def generate_slow_attacker():
    user = random.choice(ATTACKER_USERS)
    ts = random_timestamp()
    signature = 4625  # failed logon

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": "185." + str(random.randint(1,255)) + "." + str(random.randint(1,255)) + ".12",
        "dest": random.choice(WORKSTATIONS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": "failure",
        "process": "powershell.exe"
    }

def generate_password_spray():
    user = random.choice(NORMAL_USERS + ADMINS + SERVICE_ACCOUNTS)
    ts = random_timestamp()
    signature = 4625

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": "185." + str(random.randint(1,255)) + "." + str(random.randint(1,255)) + ".33",
        "dest": random.choice(WORKSTATIONS + SERVERS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": "failure",
        "process": "unknown"
    }

def generate_insider_event():
    user = random.choice(NORMAL_USERS)
    ts = random_timestamp()

    signature = random.choice(list(SIGNATURES.keys()))
    action = "success"

    return {
        "TimeCreated": ts,
        "user": user,
        "src_user": user,
        "src": "172.20." + str(random.randint(1,255)) + "." + str(random.randint(1,255)),
        "dest": random.choice(SERVERS),
        "signature_id": signature,
        "signature": SIGNATURES[signature],
        "action": action,
        "process": "C:\\Windows\\System32\\cmd.exe"
    }

GENERATORS = [
    (generate_normal_user, 0.50),
    (generate_service_account, 0.25),
    (generate_admin_user, 0.10),
    (generate_slow_attacker, 0.05),
    (generate_password_spray, 0.05),
    (generate_insider_event, 0.05)
]

def weighted_choice():
    r = random.random()
    total = 0
    for gen, weight in GENERATORS:
        total += weight
        if r <= total:
            return gen
    return GENERATORS[0][0]

def main():
    with open(OUT_FILE, "w") as f:
        for _ in range(TOTAL_EVENTS):
            gen = weighted_choice()
            evt = gen()
            f.write(json.dumps(evt) + "\n")

    print(f"Generated {TOTAL_EVENTS} synthetic events → {OUT_FILE}")

if __name__ == "__main__":
    main()