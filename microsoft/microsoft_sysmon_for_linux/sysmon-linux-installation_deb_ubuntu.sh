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
<Sysmon schemaversion="4.90">
  <EventFiltering>

    <!-- ========================= -->
    <!-- Process Creation (EventID 1) -->
    <!-- ========================= -->
    <RuleGroup name="process_create" groupRelation="or">
      <ProcessCreate onmatch="include">

        <!-- Shell usage -->
        <Image condition="end with">/bin/bash</Image>
        <Image condition="end with">/bin/sh</Image>
        <Image condition="end with">/bin/dash</Image>

        <!-- Common tool transfer -->
        <Image condition="end with">curl</Image>
        <Image condition="end with">wget</Image>
        <Image condition="end with">ftpget</Image>
        <Image condition="end with">tftp</Image>

        <!-- Suspicious temp execution -->
        <Image condition="begin with">/tmp/</Image>
        <Image condition="begin with">/dev/shm/</Image>

        <!-- Account manipulation -->
        <Image condition="end with">useradd</Image>
        <Image condition="end with">adduser</Image>

        <!-- Permission changes -->
        <Image condition="end with">chmod</Image>
        <Image condition="end with">chown</Image>

      </ProcessCreate>
    </RuleGroup>

    <!-- ========================= -->
    <!-- Network Connections (EventID 3) -->
    <!-- ========================= -->
    <RuleGroup name="network_connect" groupRelation="or">
      <NetworkConnect onmatch="include">
        <Image condition="end with">curl</Image>
        <Image condition="end with">wget</Image>
        <Image condition="end with">ssh</Image>
      </NetworkConnect>
    </RuleGroup>

    <!-- ========================= -->
    <!-- Process Termination (EventID 5) -->
    <!-- ========================= -->
    <RuleGroup name="process_terminate" groupRelation="or">
      <ProcessTerminate onmatch="include" />
    </RuleGroup>

    <!-- ========================= -->
    <!-- Raw Disk Access (EventID 9) -->
    <!-- ========================= -->
    <RuleGroup name="raw_access" groupRelation="or">
      <RawAccessRead onmatch="include" />
    </RuleGroup>

    <!-- ========================= -->
    <!-- File Creation (EventID 11) -->
    <!-- ========================= -->
    <RuleGroup name="file_create" groupRelation="or">
      <FileCreate onmatch="include">

        <!-- Temp directory activity -->
        <TargetFilename condition="begin with">/tmp/</TargetFilename>

        <!-- SSH persistence -->
        <TargetFilename condition="end with">authorized_keys</TargetFilename>

        <!-- Cron persistence -->
        <TargetFilename condition="begin with">/etc/cron</TargetFilename>

        <!-- Systemd persistence -->
        <TargetFilename condition="begin with">/etc/systemd/system</TargetFilename>

      </FileCreate>
    </RuleGroup>

    <!-- ========================= -->
    <!-- File Deletion (EventID 23) -->
    <!-- ========================= -->
    <RuleGroup name="file_delete" groupRelation="or">
      <FileDelete onmatch="include" />
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