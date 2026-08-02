# Security policy

## Supported version

Security fixes target the latest release and the default branch.

## Reporting

Do not open a public issue containing a credential, exploitable prompt or trace,
or sensitive repository content. Submit a
[private security advisory](https://github.com/pranavdulepet/budget-router/security/advisories/new).

Do not transmit a live secret. Revoke or rotate it first, then retain only a
redacted reproduction.

## Credential handling

Budget Router reads provider credentials only from environment variables.
Credentials must never enter configs, traces, grader artifacts, Docker task
containers, screenshots, issue text, or commits. Any credential exposed to a
prompt or log must be treated as compromised and rotated.

Coding-agent tasks and candidate patches are untrusted input. Run terminal
evaluation only in the official isolated benchmark containers, on a host that
does not forward provider credentials or unrelated filesystem mounts.
