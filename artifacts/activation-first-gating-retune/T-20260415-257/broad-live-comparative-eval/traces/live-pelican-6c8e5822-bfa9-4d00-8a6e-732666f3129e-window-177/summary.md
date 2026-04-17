# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22c0c5cfe30f6528627aae6b3b1ce6c55137840c4388f7d03d5ba0c64043e114`
- fixture hash: `sha256-883333e2877ee56be18afd0bdb26f3a044eab5df448e40bf59cfd947e2e070a7`
- score hash: `sha256-c669370cbe367664610b2e96dbba462fdb8621dcc6a2801ee24fa35773e1e3ae`
- bundle hash: `sha256-fc2fd44b45c0a2af9d0b0de7c10d6bf1a8908d609c9a023725de9a5a3751c805`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bfacda8f5501f5e4f01bbebcdaf7a5c0e18d211755bb5803d41f576de0d46bba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c35ac8f266ffbd71423da0019b225443e3e7039d31ca430175b2045406512499 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-332357c1c89a28a4c8b62b3230ddd3b181ec59d7748944eaff5505a5963b1aee |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fab7b506695b3bd3f7faec885c03f278e9b5c123a886fc3ac83b9bea35457c42 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bd3bbe54 | sha256-5f47a79a13a6efd5e297b8822d64a451ce0107ba9a321b2f8f5a5f036d6c6675 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bd3bbe54 | sha256-8d12e122afc3ab5f4db338665dd6b3c3c1b7b710068d0a11d5bfc762b5812a1d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-af11ff05 | sha256-ee49aca0a170417e89e1a5097f3c1c8dfdb4d10c62124e57860d1b40961cf56b |
