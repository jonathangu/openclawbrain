# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1ec62e8076ee4d1e710644be210d5ded13133f83ba7cc0a283a8ff2ec6e4b13a`
- fixture hash: `sha256-208011a3d49bd10b0f228ef3f15f5d25a591b8469fe6d29ce8deec0246fbbb48`
- score hash: `sha256-71be8e12c062757578e00c13524a9e2f5ef5e9b504ef332a9bb1599fe4e9f965`
- bundle hash: `sha256-87c577eb624103f7862d4e5e7ced1f6c7dc17d5ce0a3742ec59222bfc6f339d3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30241ba1cbd874d0509ab1e29b9c021ef1eb69d9f017747456f3594de63d356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2fa311245a693b66283e0efd60b64e5472cd824f62c33d2cd6f71e47df15e50 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d36ec347daeb77cca3fe6ab2f9b8f34d4a7172b8f3fe2927d90fda3f164c7a3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ff82007b7a612c04423a8afef3d372fe53875bf7e1a7b08767e2f50622543636 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-706f6801 | sha256-33d5964909d409d8bd4f39e28964fa4f6f86d5bfb26d3f4304eb25eb25288e21 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-706f6801 | sha256-183c18de3aef9c7068bcf98970fe5e3d0077485b8901ad7aacb2c5193f52f6c4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3290b714 | sha256-08fb72384ecb2d1d9ee37f4a297359cc739ae8ac9b21e6ca9d65228d511bd712 |
