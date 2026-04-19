# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f2e67ba6e9f3ee34d9a729b960d4347b90c5776b36c8bb01215597777ac63b8`
- fixture hash: `sha256-31116913aa40fd67b6f1a05c1b62a0f72f8a386379a84cc5c256525c2b570370`
- score hash: `sha256-943ebe1b9d853e30f1b5e0ef3977b396b32de0845fcd2e0e4d543819e8fdca18`
- bundle hash: `sha256-27a2cb50086b151c460e4a6bef5f16db5b07274f0fa8795440775a0247e8a521`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13aa42d069a6fbba4caba9f912ef9cadf19ea12093ab266f931b4282b9e22bf |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-3444b169efef7c771848bb30e266d22b24cd2797e10b5bfcbba9aae92055062c |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-9a25516278f526f35317e9333225d450f1ff410f145970af6e09f2c834ccaa73 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-60836c084e803670bf250cb6a3be82d7c70a7ad5479546ee63d5df7ef0de450e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-fe581a28 | sha256-3a217717dbe0e7daa6d372a05508f5d1757c2db3aaef107a5bc1fc4769f2271a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-fe581a28 | sha256-db5b1b10ce4bb8d4e77f41b14222c14948b58229b047196e40050f807c06edbe |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-fe581a28 | sha256-3a217717dbe0e7daa6d372a05508f5d1757c2db3aaef107a5bc1fc4769f2271a |
