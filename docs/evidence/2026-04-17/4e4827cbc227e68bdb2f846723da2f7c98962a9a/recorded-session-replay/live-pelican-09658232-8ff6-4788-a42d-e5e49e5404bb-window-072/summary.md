# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ea52ae43f846ec5905bcf93ba8cbe911779b358315c7c925dfcb3bb8d88c42d`
- fixture hash: `sha256-5ca126c8a28da22685d19c74b8dc7e5cb0bac37c0b916d2162c68f83275f6394`
- score hash: `sha256-d81ea80603f82108cadc5d01110792490017585c39250fb364e73b769d45d4cc`
- bundle hash: `sha256-6dba59473e82077c8a5099acc58e5fe2947e933e4f18b0ce91d940b1e28377e1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ee8e31cf8824d425cda16aa09e9361fc2028a17b7c4fcbcc21c2fa64f147edf |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10d817f4f531a58bdb90e47fb0402a1b364096fab17c224a3221112fb704e4f2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6441dba4204d3dab8e49cff07ee0d38efd60d07e6fbb219e4c37f69826cd68cf |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8049969df50c6c7fe8e3c1a7f07433afb10846984d954e34b04e9ee486d70ff5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b0199b4 | sha256-fa89d1a26c7bda3ecb942f44646bec129580ff1d91e2fab9ac30cc9392f7434f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b0199b4 | sha256-a84942603b234d1bfe4cda446f2dedeb803f4c9d785b11e49badf5fd7bb91836 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4ad1cd89 | sha256-dd89d5ac5937a19f517f871000c3fd0262f3cb1954b685b564c142793d7529e6 |
