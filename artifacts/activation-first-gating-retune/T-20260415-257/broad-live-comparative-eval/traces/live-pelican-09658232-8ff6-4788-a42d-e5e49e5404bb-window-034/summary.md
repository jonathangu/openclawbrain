# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-12ec3556496594cbfe1c9babd4c4adcee8cc4de6490ecd2d9fbb210c45663369`
- bundle hash: `sha256-a83a1e9314a000824903bd7a973294dcca82fccea2f42160528bec5cd6fdeb4f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-06c32f3b5e39b54ac1e19f16b4910237ff4dcc611d2ed9a2a46fac40d71a4751 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-27a6ad5be36d5f3c9d67ca6ca0787edabe5d2e8bf1cff26aeca2bdd5035a4ebd |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6ee05471e16e66bcedbda70ddac9452f6458693fa60c40e02ad6f5b80946378f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-90366f07 | sha256-018693534b17b26906fc38b9bc323e0e10d3c16a586de75e668ee89e14e7b57e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-90366f07 | sha256-6f4072045de114d6ef77201944a111a31e030def7fa7614fecc6edacb8aef414 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-90366f07 | sha256-018693534b17b26906fc38b9bc323e0e10d3c16a586de75e668ee89e14e7b57e |
