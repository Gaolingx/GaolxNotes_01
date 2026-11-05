# Nginx vs IIS vs Apache 对比总结

## 🧑‍💻 User

nginx vs iis vs apache

---
## 😀 默认助手 (7)

# Nginx vs IIS vs Apache

These three dominate the web server space but solve the problem in fundamentally different ways. The short version: **Nginx wins on concurrency and efficiency, Apache wins on flexibility and ecosystem, IIS wins on Windows/.NET integration.**

## Architecture at a Glance

| | **Nginx** | **Apache (httpd)** | **IIS** |
|---|---|---|---|
| Concurrency model | Event-driven, async, single-threaded workers | Multi-process, thread-per-connection (`prefork`/`worker`/`event`) | Thread-based with kernel I/O completion ports |
| Process model | Master + N workers (N ≈ CPU cores) | Master + child processes, each handling connections | `svchost`/`w3wp` worker processes, app pools |
| Memory per connection | Very low (~few KB) | High in `prefork` (~MBs/conn), lower in `event` | Moderate |
| Dynamic content | Proxy to PHP-FPM, uWSGI, etc. | Embedded modules (mod_php) or proxy | Native via ASP.NET Core / FastCGI |
| Config style | Declarative nginx.conf blocks | Declarative `httpd.conf` / `.htaccess` | GUI (IIS Manager) + `web.config` XML |
| Platform | Linux, BSD, macOS, Windows (limited) | Linux, BSD, Windows, macOS | Windows only |
| License / cost | BSD-like, free | Apache 2.0, free | Bundled with Windows Server (paid OS) |

## Performance & Scaling

- **Nginx** — Designed for C10K+ problem. Non-blocking event loop keeps memory flat as concurrent connections grow. Excels at static files, TLS termination, reverse proxying and load balancing. Often 2–4× the requests/sec of `prefork` Apache on the same hardware for static workloads.
- **Apache** — `prefork` (process-per-request) collapses under high concurrency due to memory. The `event` MPM narrows the gap significantly, but embedded `mod_php` is inherently process-bound. Still fine for moderate traffic, especially with PHP-FPM behind it.
- **IIS** — Strong on Windows for ASP.NET / ASP.NET Core. Historically slower for massive static-file concurrency, but modern IIS 10 + HTTP/2 + output caching is competitive. Kernel-mode `http.sys` gives excellent HTTP parsing and caching.

## Configuration & Flexibility

```nginx
# Nginx: server/location blocks, inheritance rules
server {
    listen 443 ssl http2;
    server_name example.com;
    root /var/www/site;
    location /api/ {
        proxy_pass http://127.0.0.1:3000;
    }
}
```

```apache
# Apache: directives + .htaccess (per-directory override, no reload needed)
<VirtualHost *:443>
    ServerName example.com
    DocumentRoot /var/www/site
    ProxyPass /api/ http://127.0.0.1:3000/
</VirtualHost>
```

- **Apache's `.htaccess`** is its killer feature and its biggest liability: drop-in per-directory config without touching the main config (great for shared hosting), but it's re-parsed per request and hurts performance.
- **Nginx** has no per-directory override — all config is central, which is faster and more predictable but less delegation-friendly. Its module ecosystem is largely **compile-time**, requiring rebuilds for third-party modules (or dynamic modules in newer versions).
- **IIS** is the most GUI-driven; `web.config` inherits and can be overridden per-app, which maps naturally to multi-tenant ASP.NET hosting.

<details>
<summary><strong>Feature matrix (click to expand)</strong></summary>

| Feature | Nginx | Apache | IIS |
|---|---|---|---|
| HTTP/2 | ✅ | ✅ (2.4.17+) | ✅ (Win10/2016+) |
| HTTP/3 / QUIC | ✅ (1.25+) | ⚠️ limited/experimental | ✅ (Win11/2022+) |
| Reverse proxy / LB | ✅ first-class | ✅ via mod_proxy | ✅ via ARR |
| Built-in caching | ✅ fastcgi/proxy cache | ⚠️ mod_cache | ✅ output caching |
| Per-directory config | ❌ | ✅ `.htaccess` | ✅ `web.config` |
| URL rewriting | ✅ `rewrite` | ✅ mod_rewrite | ✅ URL Rewrite module |
| gRPC | ✅ (1.13+) | ⚠️ partial | ⚠️ via HTTP/2 |
| Windows auth (AD/Kerberos) | ❌ | ⚠️ plugin | ✅ native, best-in-class |
| Let's Encrypt / ACME | ✅ certbot/cert-manager | ✅ certbot | ✅ win-acme |
| Shared hosting friendly | ⚠️ | ✅ | ✅ |

</details>

## When to Choose Which

**Pick Nginx when:**
- You need a reverse proxy, API gateway, or load balancer in front of app servers.
- Static asset serving, TLS termination, or very high connection counts.
- Container/Kubernetes environments (ingress controllers, sidecars).
- You want low memory footprint and predictable scaling.

**Pick Apache when:**
- You rely on `.htaccess` (shared hosting, WordPress plugins that need it).
- You need a specific Apache module (mod_rewrite edge cases, mod_security, mod_dav).
- Legacy PHP apps with embedded `mod_php` and simple deployment.
- You want the broadest documentation and community answers.

**Pick IIS when:**
- You're on Windows Server and running ASP.NET / ASP.NET Core / WCF.
- You need tight Active Directory, Kerberos, or Windows authentication integration.
- Enterprise environments managed via PowerShell, Group Policy, and IIS Manager.
- Hosting legacy .NET Framework apps that assume Windows-only APIs.

## Common Real-World Recipes

| Scenario | Recommended stack |
|---|---|
| WordPress at scale | **Nginx** → PHP-FPM, or Apache behind Nginx |
| Node/Python/Go APIs | **Nginx** as reverse proxy + TLS |
| Corporate .NET intranet | **IIS** + Windows auth + ARR |
| Shared hosting / cPanel | **Apache** (or LiteSpeed) |
| Kubernetes ingress | **Nginx** (Ingress-NGINX) |
| High-traffic media/static | **Nginx** + CDN |
| Legacy PHP + `.htaccess` plugins | **Apache** (+ optional Nginx front) |

## Market Share (rough, 2024–2025)

Nginx leads active sites (~30–35%), Apache second (~25–30%), IIS third (~5–10%) but **dominant within Windows-hosted and enterprise .NET estates**. The gap is smaller than headline numbers suggest because "active sites" skew toward Nginx/CDN-fronted deployments.

## Bottom Line

- **Nginx** for performance, proxying, and modern cloud-native architecture.
- **Apache** for compatibility, per-directory config, and module-heavy legacy stacks.
- **IIS** for anything Microsoft — it's not really competing on raw benchmarks, it's competing on native Windows integration.

A very common production pattern is **Nginx in front, Apache or IIS behind** — Nginx handles TLS, caching, and connection management while the origin server handles app-specific logic.
