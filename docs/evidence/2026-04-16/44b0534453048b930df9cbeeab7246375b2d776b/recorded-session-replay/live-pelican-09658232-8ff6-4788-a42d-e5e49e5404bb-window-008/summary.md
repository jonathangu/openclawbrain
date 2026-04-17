# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc7aa6c27637d299d6eae706b4fc67a2a2a7b4de77c818a562317ac57ca7ac6f`
- fixture hash: `sha256-3d9a8c7638fdfa743ac7a63700e6bcceed5b6728eed1bfa78f1b2db0ab28c6de`
- score hash: `sha256-d51f81f7f117939a5ecb626c3dd1ef33d9bc998f23958538a6d28a5052029a71`
- bundle hash: `sha256-9ce1288baec5d172bf7766abaa912932b0a8f91eaf8a33e40e01c655c485de80`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6474375f5bcb6a5860753785382ca496af4bf19e7ca31262302583c0776eda20 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a57aea59e336cad2b850ad189159833f159b80c12cde06afd805cfc0317a72b6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7b12f43467c45df5c4cac932a7669c2a72f401092f5077d0b179616529730c62 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-690c90c9962157ad606b9cfbe6559167cbc76738028dbf68672582e36b350841 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7961e370 | sha256-5d09dcc03c4ad63473bd487484cdff8341113f9359aa6dc0d56aefc127bd57af |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7961e370 | sha256-5295f167c202399604097bbe3ac389ea83015a926480bb34aa060924c1508a84 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-93e0535d | sha256-86e515aa3b6c02a7abc27035cf9b211beae8105f583cb502bdff24a22ed2dec6 |
