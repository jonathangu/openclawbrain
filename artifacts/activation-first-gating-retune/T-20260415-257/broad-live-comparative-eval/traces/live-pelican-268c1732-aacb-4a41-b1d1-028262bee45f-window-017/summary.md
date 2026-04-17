# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-41fe4ec7878578e538ea87217d0f1ff26ff7c1c5009495fddc6dab6258a2bbbb`
- fixture hash: `sha256-49ec3fe73495575ef4a5edbb2b2c58d86b67a86df5a7ca6830045265a7717b0f`
- score hash: `sha256-cf8b761fa83b30b2fb155b6f94274e6a62da5ff1c700d20aecf8001aae2d364c`
- bundle hash: `sha256-3516824ab3e4ad1248867afcbc98f224493c00f5bdfc1696aad186e0d6a033bd`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5a96d1404eefca9da775b3bd1f6864e8e794d06c6c90d16aa5e90455db3aa7d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-458ba989895e813d97648a7ea923ee7d0ab633e4114a5ea30b1f312902314c85 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39081bcb14c935a823f55defd8af62ebba2abd58d2e3519282739760c6a4b691 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-06c9a84eb7fb1871708ca89782b9ec24c117467ffbb81c72320a290cdefcdfba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-de3e663a | sha256-11ec2c1238ff948621629ae7645b56e235e17867f8114784daaefeb7321e14de |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-de3e663a | sha256-49886c84127d6baafb1b77e58b41bde1bb9e4d04161883c5e7834906f36d936f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1e33c14f | sha256-ea8445e1599f4d91e049223d2b1006a399889b471659e5c8e6416495f1a02619 |
