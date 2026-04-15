# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84b9a4843680de911479c2420a8592984c3d84b3d54d06debdc96d5c918ea030`
- fixture hash: `sha256-be373dad3e692162d5000f12580f9371232c68a9b0f09d3136130b3fe2a640e9`
- score hash: `sha256-275d97bf56ee87f2abec7bc4887c1b4bf8b5e7f70239eba75b43232b8f1559be`
- bundle hash: `sha256-e37a69006917baa9c8bc2d2dfcd2fe9d312a28d31a86b5ef78f49a4adc39691c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-34864147a65f338d5fe87baff27e70ea8462feed84ac2fbd4644ab5e3e006364 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ba98c6d1c8b88061895f87259bdabe447eb4ba9b3612f017bed3d7f2cdc5c941 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-40257ce8c1713e2d94fab96456dc5ea52f5b451d41723e8156b9d3a2b18ec055 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-036222697352d2b6b6bbcd1e96363bc72544a1635296e658454536eb1642e6d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dff0f340 | sha256-9ea486e67fc533b899cc410b7b744c3b6dac0cd64e39d3e969c408ffcf9788f5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dff0f340 | sha256-04abaaebb4cd1f828a657b1779c1668e63c0c3d0c8b3a99184884eaa0a7869d3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-dff0f340 | sha256-9ea486e67fc533b899cc410b7b744c3b6dac0cd64e39d3e969c408ffcf9788f5 |
