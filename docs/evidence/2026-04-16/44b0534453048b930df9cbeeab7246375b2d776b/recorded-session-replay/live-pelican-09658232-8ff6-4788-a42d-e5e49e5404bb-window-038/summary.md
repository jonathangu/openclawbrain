# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb85648e6a75ae5a3ba6cc73943d9146864f7df475704ace40a5204fa142526`
- fixture hash: `sha256-7f13329cf1857fada1958a6fb5e614617a7842e6a6185ecb6a9d160264a3397f`
- score hash: `sha256-c78ba9158ca302595040b37221bb9af3a21679c1818fb4079804d142d9a0dbed`
- bundle hash: `sha256-e805471f9b52ac5590b6e8c95804ed2aab9df0c8b50e8bd0ccbc8707051459c8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8dfb1b133ecf1f6249a4f0b0aded2fd5af80368793a13e3eb702ebb8c1e8fca |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a1eac50b709fb03b83add96767088f9291d9670c4b79264608ab8426950cb56b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ccf954e63ef3b2aebf0770cb676d54d9a5a588c583862a1e7d24735d91680615 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8ecb56aa6f2aa96efd3ea453686aa169de78e1f5e1363fe3ac9fe7194a099eaf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1c4bc8c6 | sha256-36764148291bd11b29d2bce056f466f6974d194eaf054a787b511e6a359e8b70 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1c4bc8c6 | sha256-911560172db73a4b01ba89c540bf418e9e1211a51f4f5813b8d8b1d91c045703 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-53f3190f | sha256-997355fae82504e0536fdfcc6f504ea11480b49ded8d63c54fec02d5ae33b6e3 |
