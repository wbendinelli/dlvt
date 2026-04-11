# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes     |
| 1.x     | ❌ No (upgrade to 2.x) |

## Scope

**dlvt** is a scientific Python package for numerical simulation of a dynamical systems model. It performs no network I/O, handles no user authentication, processes no sensitive data, and has no external service integrations.

Security concerns are therefore limited to:
- Dependency vulnerabilities (NumPy, SciPy, Matplotlib)
- Malicious inputs to simulation functions that could cause unexpected behaviour

## Reporting a Vulnerability

If you discover a security issue, please **do not open a public GitHub issue**.

Report vulnerabilities privately via email:

**wbendinelli@gmail.com**

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

I will acknowledge the report within **72 hours** and aim to release a patch within **14 days** for confirmed issues.

## Dependency Security

This package depends on:
- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`

Keep dependencies up to date to minimize exposure to upstream CVEs.  
You can audit your environment with:

```bash
pip audit
# or
safety check
```

## Disclosure Policy

I follow responsible disclosure. Once a fix is released, the vulnerability will be documented in [CHANGELOG.md](CHANGELOG.md).
