# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a95260c17a69374ef7a9ff20490cb415b09868b4babdf035ba541b6d82beb5bf`
- fixture hash: `sha256-ff74599acd0d3d5ad2046fb7795a787fe8fa0e70837c98ae65f89838fc9f50e9`
- score hash: `sha256-cbcb5badf060378efc1bc645d8c4311c5f4b8d50e3db79aa5fa5edd43f3f81f9`
- bundle hash: `sha256-3514183ffc4ebd6ee2291383fbbb63960245187af77e7999be551c1e64ed4f51`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30f22d5abba84d169a9ef0f72b28eb7bd4c2afa26a7910c928c371f416decf04 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11ac121c4b9ff997de236f996e7f47c6d17a0af92b5b3b52817f31e2f8749dab |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c369b4eba8fd690bc9bd70500c35fcf7f1d792d160b32744dcfaf3e66289237d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-af6541fd2fb8efe7d4c60fd10c422077023d120ad554778f16fef181ac8eec7b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9e701bd | sha256-6f1627fd776cdef28542f433d7192a84064cff635c853f6fb9548d03e695b953 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9e701bd | sha256-ae9b755d11cdcfdcfcbdaac92054b896304a228c944138098aaa4553c036dcdc |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b6bc0f76 | sha256-d10e0f194a4f41174748038d40c4647ac1c5b3ee11d6edf36f4e12406d4d006a |
