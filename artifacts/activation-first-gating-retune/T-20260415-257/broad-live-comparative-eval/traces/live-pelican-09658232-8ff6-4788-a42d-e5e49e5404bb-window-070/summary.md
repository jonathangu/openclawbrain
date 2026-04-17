# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af0623cd896f3d36aa832764b91c449eb65a56e502af4829ad2995082aa19cee`
- fixture hash: `sha256-729b2a143706d45b443dec7a409dfdba222ee805edd97aecb9fe78e30ae910a9`
- score hash: `sha256-048f04347f72765464809130c2211df67c73f5258137dc1aedd973ff3fe27156`
- bundle hash: `sha256-b3e2ff219ef5c24033186b97e97cfaced1d4b8d6e5b0e8eb6c220ce5313efd85`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72dc7dab3dc434226257b098b5889b33f6d9a175c84b5a7ecf9e06dde7b7bf77 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a81b0969297b0bfc9447bb4ae6f40b6c26eed4330ad46b486981c8186951e96d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-667317daa969940ea00ef5937971d40856b183822ba071cf143952b48194a788 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-05c0c8c14e56fe2778b955fd3d7c18bc15cb7b8aac77defc49548764a52da714 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-64a7b149 | sha256-3d49272624842a0175a0499d1697762d16b004b365982b26937d2b5afe4530ac |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-64a7b149 | sha256-82ac1ac8b31d9e21c391a081e3a4ca1f96dc2dd8d381b352e06e493105b857c7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8f117356 | sha256-1860dd896f1456454abe79631304bf79f8355277689cfa2a9330967203e544dd |
