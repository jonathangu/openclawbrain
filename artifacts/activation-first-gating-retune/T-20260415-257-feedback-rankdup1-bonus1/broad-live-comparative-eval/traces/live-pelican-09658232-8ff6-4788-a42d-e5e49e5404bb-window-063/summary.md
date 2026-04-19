# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb64d18c7380c29a36adda7f18b9d94028bd2ec79c3f043249c311ff96079b77`
- fixture hash: `sha256-4d9c945d16c80ffc64625c9921a10c1aa73d0e2d0d7dc96750c287fa87ef0a3c`
- score hash: `sha256-9c87775378c4fefd739e0101e3cb2220494a0ee192c67cc4e4ee6b507326fae7`
- bundle hash: `sha256-2971bf151a429432ddd7eeeb58844b5b7c00e217f1c0fa191fab9c432be86f01`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2057eeb74350d5472d7a207a6cd23d83fdfc1cbff7a9da70502d2c9709cf85fb |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-89f221e6fb3400819a16f0ebb9889492fae6cae8a898e0dfebf194029d1276bd |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-e59ac371bce1e5b9eb52c158f4195e97ae21ed712626a818caa715741397fc79 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6f6da50b8f40ebf02b37018b0e9380fd866d2c51b5d99b58910703f8e2de2733 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-40b01de3 | sha256-0578c26b27b3a10c884feeb08178c89da94304c4a23088f4c2aa1c67bbf70931 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-40b01de3 | sha256-1931618b1d24e3bdab0e370bf629c133ac7b1e2b2e2102bb9094edecf01a8147 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-40b01de3 | sha256-0578c26b27b3a10c884feeb08178c89da94304c4a23088f4c2aa1c67bbf70931 |
