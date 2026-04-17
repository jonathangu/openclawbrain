# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24534757347a93b75b187ac38d7e9e86602b361f9d21c2720eadd2aac5437955`
- fixture hash: `sha256-599fe6907f3cd26dea75cb20dba6e419b550fd93a91244dc2a42f5a954807c1f`
- score hash: `sha256-3869501d75bc1020f7d618a9b4a1aeaa8cc6eba6d03dadf14a1f74b7ee896a33`
- bundle hash: `sha256-76d60e183205b7f74f8b4a6450a39d8486f1b380fd6438ebb4c30b266d719fe4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fdc42d0f374b0d99f1d90ad1240350cf6baad53baa8fca8aa6efbbc417c89845 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f5e8ae130f52d63c1738d3bd04c7049d399cb7086e1028ad28afdf2c8e279b30 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f54d0c9f7e843837c340d6b173f53a7e4ffb1d408f2beb19201b4ce661c9eb5e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a33dcf67e686dd3c0ec4e53ae6665667411cdf3a37b80d281c034fab03b75263 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08803b6d | sha256-58b5edd7e68af917a8393e0e2d5cd43f291221a3f0021228b45298fa60e9f174 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08803b6d | sha256-b8a8a1fe2ce78c57a2e20553e29aa29431508aa14eeb29366e261b7bb7af7c9b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-44896b34 | sha256-c270a1556317c6e7fcc80c93e0390b0c453223ede0867b4b5307b071891560d0 |
