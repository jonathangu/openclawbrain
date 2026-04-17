# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7a2236f637704fe149867dcf144d671dc7a13fa94e04f98252bb7a94efde6a70`
- fixture hash: `sha256-3dd95fcccf0fb105acb53dbd74c41b44d30300251f8ca1b0c6b6f7ee328de982`
- score hash: `sha256-43256cb9c2617cf01131e1b735eae6c26546f701015690f9504001777ceb77a4`
- bundle hash: `sha256-3e369668c66c98348208d8910a88dd62994ab4cbe2bc6e486043a54fb57135b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef52e9f6d08b86a0755671620744d8fa71177a56d88b43c65d023da00ed4b3db |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cb1e0af92f7e6d648f8bc96aae0c90c9a3eabf8e11b7fc7b3e8112c5db0fc592 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ce244bd52632428903e8268d236e72ded599f6fd17f64ace85f15fce7f51e6e6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-47d5281942096d5071571a453e16f54dfdbfad0b43717d9d47dd5883bd945ced |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e540b2c7 | sha256-4073224cb43a8e7e052eaed917238d5de02b7c90c7706f8d2bf7a12c30c30baf |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e540b2c7 | sha256-69d66c2279e3ab217b040b83a0a2867fae0773348c218926c6fef2d007c1186f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8278b032 | sha256-8dd5e5a0e78c46ad11ef1cf5a0d089dbb58b9f499a185ff4d7722381c4211d58 |
