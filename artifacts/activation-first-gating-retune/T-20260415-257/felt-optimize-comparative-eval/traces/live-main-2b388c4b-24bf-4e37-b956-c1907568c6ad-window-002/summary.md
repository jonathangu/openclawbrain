# Recorded Session Replay Proof Bundle

- trace id: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7dfee8569cbeebae47062b33f26bd559b44ab7e32ac7a65f3d53fcd4f9d6446`
- fixture hash: `sha256-bf2c49e43d0148934d94e443780f19f84be1befb9f46554500ee32090d69fd0f`
- score hash: `sha256-d27cf9cf246688c329ed06e9441a423adb466e5bbdf9b8e0120adbd2069cc727`
- bundle hash: `sha256-b83bd0659e8f08f4f189f44bc253aa7d1ee6a811d77d3a6aa079c7bde35558a8`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aefc644f475f6e64faeecc10e1bad33424cc557b74533b3b9b16e76adc362925 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ad8e9876792ae6331da8841e52bed364fa4c762dc17b4317855220b429460014 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7d0772135a51930e79e254e9541d62bf50c12f89e3794cf2282a82fd8f27efcd |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1a888109a79638b933b91d316b19933de7aed5d157bb12af49cfe7083e59489d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aea9e734 | sha256-c1de8efcc557a24853a3c660c640e9e1ff14a7f207380cd970c6067b22f7df9b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-aea9e734 | sha256-0cf034632b10e8ac2f0af8b82e1db26ceb4c55b6d7813d071738a54b553f83e0 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-015e3fe5 | sha256-e1d4f719186655cf8d84eb7b257bc6d5c2e93ba96d7f5096d4a6d144e30c9670 |
