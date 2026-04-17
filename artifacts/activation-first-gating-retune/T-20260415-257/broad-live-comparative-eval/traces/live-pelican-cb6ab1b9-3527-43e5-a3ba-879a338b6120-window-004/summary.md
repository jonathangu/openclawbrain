# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f94a31a75f6f674deb4ed72bb4e73c45b90c17561480d33d8d146b93540cfdaf`
- fixture hash: `sha256-55ad28e1e1c0e357b90d71c5a61455a338c2e0a4ef3a7f6c092d3616039ed272`
- score hash: `sha256-ab4b5705cbb837f3e1688f8eb97709022ffc869c457cbdfa2f360115c425e600`
- bundle hash: `sha256-32d38b3e71d05f69bf198955bd52c371709358e88eee9670690015ec7c70ec7a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a7e02cf5e88271092f868ed8daefe51bb787b99a9e0166c0444d9f0e9eabb76 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a2ba857bbfef01b3addc7361b3f6f1fe7b70ac125642e20adc1c5c39e50d5d8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-debf04baf5fb4d091b4a759e6edb55ce3342cbfe6d42cb1401638dbeba8d449f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0203368211333a705321183a2bc4f7dc85d2905da856590ce7b11da2e07875c7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-87beaffe | sha256-1471b806a859fb2d776b9e0dfdace93d671667ebea9cce84d091cf29c92f306c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-87beaffe | sha256-4f42d4c3183869799aa9ef460b450a786fc24f529b2084b0730489823b438323 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-635f1cc3 | sha256-9bbcd0858d52e7534faaa2efcf972c0b6419319662d10a0c4d5d28b898d1e324 |
