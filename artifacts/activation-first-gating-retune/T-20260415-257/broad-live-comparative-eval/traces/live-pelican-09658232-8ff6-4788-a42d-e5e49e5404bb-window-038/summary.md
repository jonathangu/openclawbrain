# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb85648e6a75ae5a3ba6cc73943d9146864f7df475704ace40a5204fa142526`
- fixture hash: `sha256-7f13329cf1857fada1958a6fb5e614617a7842e6a6185ecb6a9d160264a3397f`
- score hash: `sha256-062ffe1873de1a88e422cf3d3d640255d7b9f26710724ee2f080f30d52331aa8`
- bundle hash: `sha256-093117f5eaed0f521918e51681ab150b110893eed0a30c3ab334a404d921af3c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8dfb1b133ecf1f6249a4f0b0aded2fd5af80368793a13e3eb702ebb8c1e8fca |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5ddd64694031c2736582aafccbeca92d73023bc6fa43152580c5348954ea189a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5478113405fa03b574eff574c1ffb385996e216dd5f8b256b9eaa2acdb429c89 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d0103b482675a46781ed2aa2f8eff5b3937a3813e4780db691d522dc1afc5e61 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-456ae8cf | sha256-4c026fcdad882f44bd28da777ebbca3acb6372be84a5028f4b480408f651b053 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-456ae8cf | sha256-7889ae7b2f8ca93bcf8b7cdffd2c87de135d3cca6fb26a369f478c4ea63051de |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7d123918 | sha256-7d4bf25eb25f089070b20b4419c66431f62404e761c13e86b2f01f5368325c5a |
