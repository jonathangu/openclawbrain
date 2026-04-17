# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5ab4245c094a4283a2fe623f159ada638f3a2335ec988d52906db167e4a412cf`
- fixture hash: `sha256-200539ec6ee07f9053b46fde1430980f62e83407874931f79115b5f9bd8b8337`
- score hash: `sha256-ed3f9a9a111518ab860836a83019c8a0514d04cba7a19e9010c7540a2741cecd`
- bundle hash: `sha256-3e2619fc34a54d8afe9edd02c704b6730cfdfaecc6424d9af65efbd96f00aac6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1ac429de4ad281ade866e710d7bcaf6542300ac52809bbbdfe005490548973 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6cdacc78fde344e844181bdf1a014b991a503522645d9c26754da7769b51f88c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e57724c2981168218abcfe9c7b253eb0e4d04396687deeb02453a93ed58799e8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b3e071531b69b73500a9368652b7b20b71a819dbb92a18e6453634159b5be8bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b63268f | sha256-aef87d5fb0d5f4970560a2bfa666b1eedc3602b48339fcb7cd1b061d111f079e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b63268f | sha256-a1cbbe7b9c6f39ea7a2560b8a559b7f90414078506b2f8a192b61aa0dca2bd52 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b8cca2e0 | sha256-fa9248f8e20590a0098a819fdd6525e8a1c8b46b8dae16fc8b39066f996ffe92 |
