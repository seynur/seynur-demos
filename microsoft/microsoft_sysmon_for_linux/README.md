# 🧠 Sysmon for Linux → Splunk Pipeline

A simple and practical setup to collect **Sysmon for Linux telemetry** and send it to **Splunk** in a structured, CIM-friendly way.

This repo provides both:
- ⚡ One-command installation (script)
- 🔧 Step-by-step manual setup (via blog)

---

## 🚀 Quick Start (Run & Done)

```bash
curl -sSL https://github.com/seynur/seynur-demos/microsoft/microsoft_sysmon_for_linux/sysmon-linux-installation_deb_ubuntu.sh | sudo bash
```

---

## 🧩 Architecture

```
Sysmon → rsyslog → /var/log/sysmon.log → Splunk Forwarder → Splunk
```

---

## 📦 Repository Structure

```
.
├── sysmon-linux-installation_deb_ubuntu.sh
├── config/
│   └── sysmon.xml
├── oyku_linux_sysmon_props/
|   ├── local/
│   |   ├── app.py
│   |   ├── eventtypes.py  
│   |   ├── props.py   
│   |   ├── tags.py
│   |   └── transforms.py   
|   └── metadata/
|       └── local.meta
└── README.md
```

---

## 🔍 Validation

```bash
systemctl status sysmon
journalctl -u sysmon -n 5 --no-pager
```

```bash
tail -f /var/log/sysmon.log
```


## 📡 Splunk Setup

### 1. Install required parsing app

⚠️ To properly parse and normalize Sysmon events, install the following app on your Splunk environment:

👉 **oyku_linux_sysmon_props**

Place it under:
```
$SPLUNK_HOME/etc/apps/
```

Then restart Splunk:
```bash
splunk restart
```

---

### 2. Configure data input (on forwarder)

```bash
splunk add monitor /var/log/sysmon.log -sourcetype linux:sysmon -index osnix
splunk add forward-server <SPLUNK_IP>:9997
```

---

### 3. Verify data

Search in Splunk:

```
index = osnix sourcetype = linux:sysmon
```

---

## 🛠️ Supported Systems

- Ubuntu 18 / 20 / 22 / 24  
- Debian 12  
- x86_64 only  

---

## 🎯 Goal

✔ Capture Linux activity  
✔ Normalize logs  
✔ Make them searchable in Splunk  
✔ Enable security use-cases  
