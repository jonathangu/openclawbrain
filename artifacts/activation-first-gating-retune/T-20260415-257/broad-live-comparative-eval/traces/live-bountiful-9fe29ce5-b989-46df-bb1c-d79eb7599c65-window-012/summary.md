# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8cad3f46eed815106d3bcaa6e251d3bdcbd2748a5a2e05af3c4a013db9a57004`
- fixture hash: `sha256-0dbe5f4100592cfe93341d37b8f8c029b314e4248749094aab73a6cddcec834b`
- score hash: `sha256-022caf7dbe13c1f28d09aab1e5e0dab52ff7d52d6fec5d34d341feef6a4b6314`
- bundle hash: `sha256-e95120f3a9cdcfff2a8178f691e3ba8c33f09eea976e77df14e13ae1bc208e6a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-70b89b4c1dec935305e4546f6cef47f2e70c9b2ff0e3d82f19d8e936c0d67142 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d7c9df004214fb0483b8142072ad9c925387ed069465b688beafa22216d4d8a5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-35fd4d11eef54be91ff7adbb297dd77e568f6bfbbe7eccfb277a9759f2144eee |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4cae7cec677436dbab92769159bdf1dda7a78feb973178f91f13258324cd4ab5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd247930 | sha256-1e3bf147517dd5f45ce15df9ac5b8a7f5176baec9b306c5075220413ee69a252 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd247930 | sha256-1f1dd3a40a9f4986dae1aefaaf3f08399623d67daefa250a982f7a61d32e035a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-638c601b | sha256-5bc9553843803557b3beef75ad637ea612fe15e8089993d227a04d83f7be395f |
