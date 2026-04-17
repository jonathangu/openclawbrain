# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-04fe85c8a179be229d8c68dec97a25113ffce0ef409792233e0ccc1c65106721`
- fixture hash: `sha256-66a77cd573b5398a7b3b4867686fe20ef718501f851c3ff410c457c68968fa97`
- score hash: `sha256-b113a8ac7be3555d82a91900f97df63604ccef6da31d98cd7f9e2fada5679133`
- bundle hash: `sha256-133f9a3117f8768d279f91ba28919942bfb3e459895e47c74904b75c66651fd9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae930e12c21c7056f67f547427d9cdedef7d7970b442aa81b3fdb75182425c80 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1de74360442afc8e44abf26761b35cd74f9ccc17810e413b1d05fe6647585ecf |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b0e3aa0329a48fd05f5bee550db3bb027569336b183352572fd0bf46d6af101 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d018bb2fd9df98db1a379341ea20408f17cd436c61f556860333d2888cb28f4e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d2d4e8b | sha256-c4b13846783df481f4d81a3fd88efd64444252457d202a493760b8810e0b063f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d2d4e8b | sha256-1067b06f97cb44df5916f65e506f564365309c2e262edf390dd1a180698f61cb |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6aaa7e1a | sha256-86abd367a5420b74e071e9418435901e1b64ecb2d3cedb01632dd6d6a6a43be3 |
