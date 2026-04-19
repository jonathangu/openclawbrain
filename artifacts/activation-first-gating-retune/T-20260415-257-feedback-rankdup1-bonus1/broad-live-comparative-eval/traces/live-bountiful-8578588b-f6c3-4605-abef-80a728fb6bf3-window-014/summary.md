# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-428224300aac4f3f06119d7ecdc386e98266d57e94ee4a94bb3cd7cf1ec92fdf`
- fixture hash: `sha256-48b4c185a62e99db11ed2f294b118b4c1031f6e03346711368215f8f92e0f14c`
- score hash: `sha256-0feeb7de0056f8f44de4a524d0ec6cc908e78ef611c7d276582174cc3ad78c31`
- bundle hash: `sha256-ca852302b449f0219df6a21d54284f3a12aae03986ca7e6b3c9adcc07e314abd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d4933e52342083e2c61aa4dd43d6b584304761341a273e9d037973938748bea |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-14b80c00a9ffb837e5482052d57160e413a5c2030ff849e749fedbd71bc93320 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-375cd225b301cffddb0b720d97ff51d820ed495f95230e8573bda1372d3d2faa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-17729b0e432eabfbeedecf56ac30bf0b420b6795e2d790312cb766ecfc8966d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-cecc2a24 | sha256-b029cade02483238ba9951c7a0452cb476c5a8b35e3729d266c50fb0b8982171 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-cecc2a24 | sha256-6e2cbc1c8b1f95de65105ef90e699e745d299427a888baaaeaf7478095e95838 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cecc2a24 | sha256-b029cade02483238ba9951c7a0452cb476c5a8b35e3729d266c50fb0b8982171 |
