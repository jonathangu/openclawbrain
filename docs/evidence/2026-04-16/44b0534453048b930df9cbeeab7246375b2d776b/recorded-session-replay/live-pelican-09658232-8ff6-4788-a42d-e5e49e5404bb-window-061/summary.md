# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70ab1ae3977bf5b8f105672a2af7f511f5d5e8eab54227af0f4c11c32810b91e`
- fixture hash: `sha256-f43d8483c3b4eb473890c9d4aad38b8eb4a81081d719d9c58fd2752db7997c33`
- score hash: `sha256-c584d2317ab80be7a33e5369ecf0140e60c8dedcdb89433a976afea685611cc1`
- bundle hash: `sha256-ec66163ea5a9fada7587735b6754b19aea159131033374f97e240ea86b372813`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0af8593686f4dcd1625a4737259415fed87f48af0fee073ee2e87cde2bfd51e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57f29037581ff2b0f983d435e79caf623abcf516d004aebca8aaeb347469e462 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b626a6ff194d9b895a6b0314b70532ae89882634b7541eb371ce6509073b3006 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-58668f234dcc3bd118f25e611a5919be2c48194c11deb1a62f1d6b42d7b718c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01130c59 | sha256-244b687906fab43c7662abe0d17796c7faf9b75d01c556b88b3330d952fdd481 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01130c59 | sha256-0e6411df00f2988aa19c166f4da253ae67c69dc1f345de837ffc48794900c9a6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2cfd44c8 | sha256-700adaba3d7d319c98f5728faa0f4e24ba8d7488e6bb9f84d9689e10fa952483 |
