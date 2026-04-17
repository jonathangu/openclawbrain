# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22c0c5cfe30f6528627aae6b3b1ce6c55137840c4388f7d03d5ba0c64043e114`
- fixture hash: `sha256-883333e2877ee56be18afd0bdb26f3a044eab5df448e40bf59cfd947e2e070a7`
- score hash: `sha256-8fc2fd42cd0081dc57c056763304c11435c3b584eeb53d24f1ce5c54e69b0170`
- bundle hash: `sha256-6c3bf031f33801db5fc33f70855d1105e07bdc8398d77d40b46bb4b2b9b18c44`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23676bc3fe6455111c2b4ee14b204a5a1d687f22d3143aaf2312bcd185fcc8bd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74b9573a24c7ab0262389e9e0fe438ffded46b0465bc1071b3e9b2955ff2fcba |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-936587373a34bb55cb6c5a72044282b72de3baf20f7b9aa7cf57a2534b1f3db7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f4fdd9d | sha256-7e446c76ed95db44927c3fc514863f5b58d6bcb4c2e2689689150966af94b88a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f4fdd9d | sha256-badc4840bb5d5e403cb842c6dae77d5c6f576fe00790a6e17cf07d2c65b0cdbd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-01261e4e | sha256-4a869ca49a15d54248cadb270818993b763d1ce82812b40da83742828da36738 |
