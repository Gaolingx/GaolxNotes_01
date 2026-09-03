# Windows arm的SBSA要求

### 🧑‍💻 User

windows on arm must arm sbsa/sbbr?

---

### 🤖 Assistant

Based on my research, here is the detailed answer:

---

## Short Answer: **No — it depends on the category of device.**

- **Client Windows on Arm** (laptops, tablets, desktops like Surface Pro X, Snapdragon X Elite PCs) → requires **BSA + BBR** (NOT SBSA/SBBR)
- **Windows Server on Arm** (e.g., Azure VMs on Ampere Altra) → requires **SBSA + SBBR** (SystemReady SR)

---

## The Full Picture: Arm SystemReady Bands

Arm originally created **SBSA** (Server Base System Architecture, DEN0029) and **SBBR** (Server Base Boot Requirements) specifically for **Arm-based servers** under the old "ServerReady" program. This has since evolved into **Arm SystemReady**, which defines multiple "bands" for different device classes:

| SystemReady Band | Specs | Target | Windows? |
|---|---|---|---|
| **SR** (Server Ready) | **SBSA + SBBR** | Data center servers | ✅ Windows Server (Azure Ampere) |
| **ES** (Embedded Server) | BSA + BBR (SBBR or LBBR) | Edge/embedded servers | ❌ Linux/BSD only |
| **IR** (IoT Ready) | BSA + BBR (EBBR or LBBR) | IoT / embedded | ❌ |
| **LS** (Linux Servers) | BSA + BBR (LBBR) | Linux-only servers | ❌ |

### For Client Windows on Arm (what most people mean)

The **Microsoft Windows on Arm** page on Microsoft Learn explicitly links to:

> - **PC Base System Architecture (PC BSA)** — the client/PC hardware requirements
> - **Base Boot Requirements (BBR)** — the boot specification

These are the **client-oriented equivalents** of SBSA/SBBR, defined in separate Arm documents:

- `DEN0151` — **PC Base System Architecture (PC-BSA)**
- `DEN0044` — **Base Boot Requirements (BBR)** with the appropriate recipe

In other words, SBSA/SBBR are the **server-specific supplements** layered on top of the base BSA/BBR specifications. Client Windows on Arm uses the base BSA/BBR (with UEFI + ACPI), not the server SBSA/SBBR extensions.

### For Windows Server on Arm

Microsoft Azure uses **Ampere Altra** Arm-based processors for Windows 11 Arm64 VMs. These server-class platforms **do** require SBSA/SBBR compliance (SystemReady SR band), which is why generic Windows on Arm installation media can boot on SBSA/SBBR-compliant server boards or VMs.

---

## Key Takeaway

> **SBSA/SBBR are *server* specifications.** If you're building a client PC/laptop running Windows on Arm, you target **BSA + BBR** (specifically PC-BSA). If you're building an Arm server to run Windows Server, you target **SBSA + SBBR**. The underlying common requirement for all Windows on Arm is **UEFI + ACPI** (not DeviceTree).