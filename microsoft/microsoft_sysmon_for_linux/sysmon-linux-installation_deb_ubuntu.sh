#!/usr/bin/env bash
set -u

echo "=== Sysmon for Linux installation and validation started ==="

fail() {
    echo "[FAIL] $1"
    exit 1
}

ok() {
    echo "[OK] $1"
}

info() {
    echo "[INFO] $1"
}

run_cmd() {
    local desc="$1"
    shift
    echo ">>> $desc"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        fail "$desc failed"
    fi
    ok "$desc completed"
}

detect_ms_repo_url() {
    if [ ! -f /etc/os-release ]; then
        fail "/etc/os-release not found"
    fi

    . /etc/os-release

    OS_ID="${ID:-}"
    OS_VERSION_ID="${VERSION_ID:-}"

    if [ -z "$OS_ID" ] || [ -z "$OS_VERSION_ID" ]; then
        fail "Could not detect OS or version from /etc/os-release"
    fi

    case "$OS_ID" in
        ubuntu|debian)
            REPO_URL="https://packages.microsoft.com/config/${OS_ID}/${OS_VERSION_ID}/packages-microsoft-prod.deb"
            ;;
        *)
            fail "Unsupported OS: ${OS_ID}. Supported OS families: ubuntu, debian"
            ;;
    esac

    export OS_ID
    export OS_VERSION_ID
    export REPO_URL
}

echo "=== Pre-check: architecture and OS ==="
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
    fail "System architecture is $ARCH, expected x86_64"
else
    ok "System architecture is x86_64"
fi

if [ "$(id -u)" -ne 0 ]; then
    fail "Please run this script as root"
else
    ok "Running as root"
fi

detect_ms_repo_url
echo "Detected OS: ${OS_ID} ${OS_VERSION_ID}"
echo "Detected Microsoft repo URL: ${REPO_URL}"

REPO_DEB="/tmp/packages-microsoft-prod.deb"

echo
run_cmd "Updating package list" apt update
run_cmd "Installing required base packages" apt install -y wget curl gnupg apt-transport-https ca-certificates

echo "=== Validation: base packages ==="
for bin in wget curl gpg; do
    if command -v "$bin" >/dev/null 2>&1; then
        ok "$bin is installed"
    else
        fail "$bin is missing"
    fi
done

echo
cd /tmp || fail "Cannot change directory to /tmp"

echo "=== Validation: Microsoft repository URL ==="
if wget -q --spider "$REPO_URL"; then
    ok "Microsoft repository URL is reachable"
else
    fail "Microsoft repository URL is not reachable: $REPO_URL"
fi

run_cmd "Downloading Microsoft repository package" wget "$REPO_URL" -O "$REPO_DEB"
run_cmd "Installing Microsoft repository package" dpkg -i "$REPO_DEB"
run_cmd "Refreshing package list after adding Microsoft repository" apt update -y

echo "=== Validation: Microsoft repository ==="
if grep -R "packages.microsoft.com" /etc/apt/sources.list /etc/apt/sources.list.d >/dev/null 2>&1; then
    ok "Microsoft repository is configured"
else
    fail "Microsoft repository is not configured"
fi

if apt-cache policy | grep -q "packages.microsoft.com"; then
    ok "Microsoft repository is visible to APT"
else
    fail "Microsoft repository is not visible to APT"
fi

echo
echo "=== Package discovery ==="
apt-cache search sysmon || true
apt-cache search sysinternals || true
apt-cache policy sysmonforlinux || true
apt-cache policy sysinternalsebpf || true

echo "=== Validation: package availability ==="
if apt-cache policy sysmonforlinux | awk '/Candidate:/ {print $2}' | grep -vq "(none)"; then
    ok "sysmonforlinux package is available"
else
    fail "sysmonforlinux package is not available"
fi

if apt-cache policy sysinternalsebpf | awk '/Candidate:/ {print $2}' | grep -vq "(none)"; then
    ok "sysinternalsebpf package is available"
else
    fail "sysinternalsebpf package is not available"
fi

echo
run_cmd "Installing sysinternalsebpf" apt install -y sysinternalsebpf
run_cmd "Installing sysmonforlinux" apt install -y sysmonforlinux

echo "=== Validation: installed packages ==="
for pkg in sysinternalsebpf sysmonforlinux; do
    if dpkg -l | awk '{print $2}' | grep -qx "$pkg"; then
        ok "$pkg is installed"
    else
        fail "$pkg is not installed"
    fi
done

echo
run_cmd "Creating Sysmon config directory" mkdir -p /opt/sysmon

cat > /opt/sysmon/config.xml <<'EOF'
<Sysmon schemaversion="4.81">
  <EventFiltering>

    <!-- Event ID 1 == ProcessCreate -->
    <RuleGroup name="ProcessCreate" groupRelation="or">
      <ProcessCreate onmatch="include">

        <Rule name="TechniqueID=T1021.004,TechniqueName=Remote Services: SSH" groupRelation="and">
          <Image condition="end with">ssh</Image>
          <CommandLine condition="contains">ConnectTimeout=</CommandLine>
          <CommandLine condition="contains">BatchMode=yes</CommandLine>
          <CommandLine condition="contains">StrictHostKeyChecking=no</CommandLine>
          <CommandLine condition="contains any">wget;curl</CommandLine>
        </Rule>

        <Rule name="TechniqueID=T1027.001,TechniqueName=Obfuscated Files or Information: Binary Padding" groupRelation="and">
          <Image condition="is">/bin/dd</Image>
          <CommandLine condition="contains all">dd;if=</CommandLine>
        </Rule>

        <Rule name="TechniqueID=T1033,TechniqueName=System Owner/User Discovery" groupRelation="or">
          <CommandLine condition="contains">/var/run/utmp</CommandLine>
          <CommandLine condition="contains">/var/log/btmp</CommandLine>
          <CommandLine condition="contains">/var/log/wtmp</CommandLine>
        </Rule>

        <Rule name="TechniqueID=T1053.003,TechniqueName=Scheduled Task/Job: Cron" groupRelation="or">
          <Image condition="end with">crontab</Image>
        </Rule>

        <Rule name="TechniqueID=T1059.004,TechniqueName=Command and Scripting Interpreter: Unix Shell" groupRelation="or">
          <Image condition="end with">/bin/bash</Image>
          <Image condition="end with">/bin/dash</Image>
          <Image condition="end with">/bin/sh</Image>
        </Rule>

        <Rule name="TechniqueID=T1070.006,TechniqueName=Indicator Removal on Host: Timestomp" groupRelation="and">
          <Image condition="is">/bin/touch</Image>
          <CommandLine condition="contains any">-r;--reference;-t;--time</CommandLine>
        </Rule>

        <Rule name="TechniqueID=T1087.001,TechniqueName=Account Discovery: Local Account" groupRelation="or">
          <CommandLine condition="contains">/etc/passwd</CommandLine>
          <CommandLine condition="contains">/etc/sudoers</CommandLine>
        </Rule>

        <Rule name="TechniqueID=T1105,TechniqueName=Ingress Tool Transfer" groupRelation="or">
          <Image condition="end with">wget</Image>
          <Image condition="end with">curl</Image>
          <Image condition="end with">ftpget</Image>
          <Image condition="end with">tftp</Image>
          <Image condition="end with">lwp-download</Image>
        </Rule>

        <Rule name="TechniqueID=T1136.001,TechniqueName=Create Account: Local Account" groupRelation="or">
          <Image condition="end with">useradd</Image>
          <Image condition="end with">adduser</Image>
        </Rule>

        <Rule name="TechniqueID=T1485,TechniqueName=Data Destruction" groupRelation="and">
          <Image condition="is">/bin/dd</Image>
          <CommandLine condition="contains all">dd;of=;if=</CommandLine>
          <CommandLine condition="contains any">if=/dev/zero;if=/dev/null</CommandLine>
        </Rule>

      </ProcessCreate>
    </RuleGroup>

    <!-- Event ID 3 == NetworkConnect -->
    <RuleGroup name="NetworkConnect" groupRelation="or">
      <NetworkConnect onmatch="include">
        <Rule name="TechniqueID=T1105,TechniqueName=Ingress Tool Transfer" groupRelation="or">
          <Image condition="end with">wget</Image>
          <Image condition="end with">curl</Image>
          <Image condition="end with">ftpget</Image>
          <Image condition="end with">tftp</Image>
          <Image condition="end with">lwp-download</Image>
        </Rule>
      </NetworkConnect>
    </RuleGroup>

    <!-- Event ID 9 == RawAccessRead -->
    <RuleGroup name="RawAccessRead" groupRelation="or">
      <RawAccessRead onmatch="include" />
    </RuleGroup>

    <!-- Event ID 11 == FileCreate -->
    <RuleGroup name="FileCreate" groupRelation="or">
      <FileCreate onmatch="include">

        <Rule name="TechniqueID=T1037,TechniqueName=Boot or Logon Initialization Scripts" groupRelation="or">
          <TargetFilename condition="begin with">/etc/init/</TargetFilename>
          <TargetFilename condition="begin with">/etc/init.d/</TargetFilename>
          <TargetFilename condition="begin with">/etc/rc.d/</TargetFilename>
        </Rule>

        <Rule name="TechniqueID=T1053.003,TechniqueName=Scheduled Task/Job: Cron" groupRelation="or">
          <TargetFilename condition="is">/etc/cron.allow</TargetFilename>
          <TargetFilename condition="is">/etc/cron.deny</TargetFilename>
          <TargetFilename condition="is">/etc/crontab</TargetFilename>
          <TargetFilename condition="begin with">/etc/cron.d/</TargetFilename>
          <TargetFilename condition="begin with">/etc/cron.daily/</TargetFilename>
          <TargetFilename condition="begin with">/etc/cron.hourly/</TargetFilename>
          <TargetFilename condition="begin with">/etc/cron.monthly/</TargetFilename>
          <TargetFilename condition="begin with">/etc/cron.weekly/</TargetFilename>
          <TargetFilename condition="begin with">/var/spool/cron/crontabs/</TargetFilename>
        </Rule>

        <Rule name="TechniqueID=T1543.002,TechniqueName=Create or Modify System Process: Systemd Service" groupRelation="or">
          <TargetFilename condition="begin with">/etc/systemd/system</TargetFilename>
          <TargetFilename condition="begin with">/usr/lib/systemd/system</TargetFilename>
          <TargetFilename condition="begin with">/run/systemd/system/</TargetFilename>
          <TargetFilename condition="contains">/systemd/user/</TargetFilename>
        </Rule>

      </FileCreate>
    </RuleGroup>

  </EventFiltering>
</Sysmon>
EOF

echo "=== Validation: Sysmon config ==="
if [ -f /opt/sysmon/config.xml ]; then
    ok "Sysmon config file exists"
else
    fail "Sysmon config file is missing"
fi

if grep -q "<Sysmon schemaversion=" /opt/sysmon/config.xml; then
    ok "Sysmon config file looks valid"
else
    fail "Sysmon config file content is invalid"
fi

echo
if command -v sysmon >/dev/null 2>&1; then
    ok "sysmon binary is available"
    command -v sysmon
else
    fail "sysmon binary is not available"
fi

run_cmd "Installing Sysmon with provided configuration" sysmon -i /opt/sysmon/config.xml
run_cmd "Reloading systemd manager configuration" systemctl daemon-reload
run_cmd "Enabling sysmon service" systemctl enable sysmon
run_cmd "Starting sysmon service" systemctl start sysmon

echo "=== Validation: Sysmon service ==="
if systemctl list-unit-files | grep -q '^sysmon.service'; then
    ok "sysmon service is registered"
else
    fail "sysmon service is not registered"
fi

if systemctl is-enabled sysmon >/dev/null 2>&1; then
    ok "sysmon service is enabled"
else
    fail "sysmon service is not enabled"
fi

if systemctl is-active --quiet sysmon; then
    ok "sysmon service is running"
else
    systemctl --no-pager --full status sysmon || true
    fail "sysmon service is not running"
fi

echo
echo "=== Sysmon service status ==="
systemctl --no-pager --full status sysmon || true

echo
echo "=== Recent Sysmon logs ==="
journalctl -u sysmon -n 5 --no-pager || true

echo "=== Validation: Sysmon event generation ==="
if journalctl -u sysmon -n 50 --no-pager | grep -q "Linux-Sysmon"; then
    ok "Sysmon is generating events"
else
    info "No Linux-Sysmon events found yet"
fi

echo
if command -v rsyslogd >/dev/null 2>&1; then
    cat > /etc/rsyslog.d/sysmon.conf <<'EOF'
if $programname == 'sysmon' then /var/log/sysmon.log
& stop
EOF
    ok "rsyslog Sysmon configuration file created"

    run_cmd "Restarting rsyslog" systemctl restart rsyslog

    if systemctl is-active --quiet rsyslog; then
        ok "rsyslog service is running"
    else
        fail "rsyslog service is not running"
    fi
else
    info "rsyslog is not installed on this server, skipping rsyslog configuration"
fi

echo
echo "=== Final summary ==="
echo "[OK] Sysmon for Linux installation workflow completed"
echo "[INFO] OS: ${OS_ID} ${OS_VERSION_ID}"
echo "[INFO] Config file: /opt/sysmon/config.xml"
echo "[INFO] Service check: systemctl status sysmon"
echo "[INFO] Log check: journalctl -u sysmon -n 5 --no-pager"
echo "[INFO] Optional dedicated log file: /var/log/sysmon.log"