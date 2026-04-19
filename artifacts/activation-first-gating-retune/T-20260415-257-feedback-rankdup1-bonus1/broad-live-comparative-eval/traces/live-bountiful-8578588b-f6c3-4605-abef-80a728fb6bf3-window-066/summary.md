# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-63fb4cb45ad9dc0c567733233e10232470bfdf8ceab21d6358838e826794486f`
- fixture hash: `sha256-9867351ff6099d4ad5c5968b4a566b26ecb9aec41e3d4c81142c7386e19d8bf3`
- score hash: `sha256-c05ee52f277f997ac7e0aa4bd88877f614716aa4dfc1f7dde886af6f35ea0a81`
- bundle hash: `sha256-23ad0136b809189f828710cf94935d9db1dc582a6bd54982158101b0e76ad8c4`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f1e62d93a8b084333b11d3b582af82ded5069f848e9e733cc9f3aff54320eb3 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7c24fa6e459b8e57c3787ddddba901c564e5a165161c0ac87644fc3d102c0627 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-75c70c9ee0f77011caff07e941850cc6244c4a1a763c13340d9cade8cb3b71af |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5ea522015bb345a6c5a2acf4ec9a0f77c1deed62698dbe7f0c58fdccacfa2871 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2c7d5b1b | sha256-a19231f48b31efe535d95b70716d62bb764be0440f730cee7b22b547514ea134 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2c7d5b1b | sha256-b891d1e24022b58bf8f3eb9994f0f8197ba8007940f751ebdfc419d56b0d3e4b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2c7d5b1b | sha256-a19231f48b31efe535d95b70716d62bb764be0440f730cee7b22b547514ea134 |
