# Runtime and safety boundary

Echo-Bio has two separate external effects:

1. provider requests may incur cost;
2. Biomni executes model-generated Python, R or shell code on the local host.

For that reason, `echo-bio run` requires both `--confirm-paid` and `--allow-local-execution`. This
preview does not create a container or operating-system sandbox. Run it in a disposable environment,
mount only intended data, use least-privilege credentials and do not expose controlled data to a
remote model unless your governance policy permits it.

Provider calls use a fixed model and HTTPS endpoint. Responses are accepted only through the strict
provider-native action schema. Unknown submission outcomes fail closed and are never automatically
retried, because resubmission can duplicate both cost and experimental selection.

The local stores provide:

- a hash-chained trace of requests, actions and execution observations;
- content-addressed raw artifacts;
- diagnostic and scientific execution receipts;
- a non-authoritative claim–evidence graph.

These mechanisms improve inspectability. They do not prove that a statistical analysis is suitable,
that a biomedical claim is true, or that a result generalizes.
