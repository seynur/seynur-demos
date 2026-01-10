# splunk_ds_ha

This repository contains all configurations and example apps used in a Proof of Concept (POC) for building a highly available Splunk Deployment Server (DS) setup.

The goal of this repository is clarity and reproducibility, not production hardening. Every directory maps directly to a role in the POC environment.

## Repository overview

```
splunk_ds_ha/
├── README.md
├── s1_configurations/
│   ├── etc_system_local/
|   │   ├── inputs.conf
|   │   └── server.conf
|   │
│   └── etc_apps/
|       └── poc_all_deploymentclient/
|
├── ds1_configurations/
│   ├── etc_system_local/
|   │   ├── server.conf
|   │   ├── serverclass.conf
|   │   └── user-seed.conf
|   │
│   └── etc_apps/
|       ├── poc_all_search_base/
|       ├── poc_all_search_outputs/
|       └── poc_full_license_server/
|    
├── ds2_configurations/
│   ├── etc_system_local/
|   │   ├── server.conf
|   │   ├── serverclass.conf
|   │   └── user-seed.conf
|   │
│   └── etc_apps/
|       ├── poc_all_search_base/
|       ├── poc_all_search_outputs/
|       └── poc_full_license_server/
|    
└── deployment-apps/
    ├── poc_all_deploymentclient/
    ├── poc_all_indexes/
    └── poc_all_search_base/

```

Each top-level directory represents what should exist on a specific Splunk instance.

## 1. `s1_configurations/`

This directory contains configurations for the macOS Splunk instance used as a deployment client in the POC.

****Purpose****
- Acts as a managed client of the Deployment Server
- Receives apps via Deployment Server
- Used to validate app distribution and failover behavior


****Structure****
```
s1_configurations/
├── etc_system_local/
│   ├── inputs.conf
│   └── server.conf
│
└── etc_apps/
    └── poc_all_deploymentclient/
```

****Contents explained****
- `inputs.conf`: Defines basic inputs required for the POC (for example, internal logs or test inputs).
- `server.conf`: Configures this Splunk instance as a deployment client, pointing it to the Deployment Server VIP.
- `poc_all_deploymentclient (app)`: A lightweight app that contains Deployment Client–specific settings shared across all managed clients.

## 2. `ds1_configurations/`

This directory contains node-specific configurations for Deployment Server 1 (DS1).

****Purpose****
- Primary Deployment Server during normal operation
- Owns the Virtual IP (VIP) when healthy

****Structure****

```
ds1_configurations/
├── etc_system_local/
│   ├── server.conf
│   ├── serverclass.conf
│   └── user-seed.conf
│
└── etc_apps/
    ├── poc_all_search_base/
    ├── poc_all_search_outputs/
    └── poc_full_license_server/
```

`etc/system/local/` configs: 
- `server.conf`: Core Splunk server settings specific to DS1.
- `serverclass.conf`: Defines Deployment Server behavior and enables:

    ```
    [global]
    syncMode = sharedDir
    ```

    This allows DS1 and DS2 to safely share deployment state.

- `user-seed.conf`: Used to bootstrap admin credentials during first startup.


`etc/apps` configs: 
- `poc_all_search_base/`: Base search-time knowledge objects shared across all clients (field extractions, tags, basic searches).
- `poc_all_search_outputs/`: Defines outputs (for example forwarding or routing logic) used by search or forwarding components.
- `poc_full_license_server/`: Configures connection to a license server or enables full license behavior for the POC.

## 3. `ds2_configurations/`

This directory mirrors `ds1_configurations` but applies to Deployment Server 2 (DS2).

****Purpose****
- Standby Deployment Server
- Takes over automatically when DS1 fails

****Notes****
- Configuration is intentionally almost identical to DS1
- Any differences are node-specific only (hostnames, local paths, etc.)
- Shared state (apps, clients, checksums) is not stored here

Structure and contents are the same as `ds1_configurations/`.

## 4. `deployment-apps/`

This directory contains the actual apps distributed by the Deployment Server.

In the POC setup, this directory is:

- Stored on a shared filesystem
- Bind-mounted into both DS1 and DS2 at:

    ```
    /opt/splunk/etc/deployment-apps
    ```

****Structure****

```
deployment-apps/
├── poc_all_deploymentclient/
├── poc_all_indexes/
└── poc_all_search_base/
```

`etc/system/local/` configs: 
- `server.conf`: Core Splunk server settings specific to DS1.
- `serverclass.conf`: Defines Deployment Server behavior and enables:

    ```
    [global]
    syncMode = sharedDir
    ```

    This allows DS1 and DS2 to safely share deployment state.

- `user-seed.conf`: Used to bootstrap admin credentials during first startup.


`etc/apps` configs: 
- `poc_all_deploymentclient/`: Deployment Client configuration applied to all managed Splunk instances.
- `poc_all_indexes/`: Defines index configurations shared across clients.
- `poc_all_search_base/`: Search-time configurations shared across clients.

## Design principles used in this repository
- Keep node-specific config separate from shared state
- Avoid manual synchronization (no rsync, no cron)
- Prefer clarity over cleverness
- Make failover behavior observable and predictable