# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57d54d463c3d335756b9ef845ab48b21d6ba79bd455096740f5eec6ab5dcf52e`
- fixture hash: `sha256-c975a7548913ddc09f78bdcf8d6f035b2cb79bee5a8fff204c28b6e92be5b531`
- score hash: `sha256-8f8ecc06a05a46d4d0e729d0661afd511c4f2f6db537c9a88ae18edd721e7c3a`
- bundle hash: `sha256-244685bbb8795c456f798a9c0151497b8e179df8e6866839f022586828da194f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f565862470853eb1b48835f5dc58d5e78705c4b54f6971c4806d12966cc7447 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9a36c1764f0350be00ada567bc59ac1869015f72a358b7d30fafed8103ca1fa5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-06a8131fb8b844b9a15dc4d0ae716279332f0efd810f9e3c4a23749ac7ab19ca |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5581c8842573f7a1ae190c7f6fb2f3dcc36484f79ad7d39a62eddf38bb9979ed |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fdd8d85f | sha256-02aac40ac50f869759a3f7b821eb9b5ab66aaf3d8f302c6370668faf91e77933 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fdd8d85f | sha256-01958f0df96df58c2356703e346b7f405d04e97b45427b2ea0cf2d2afa233626 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-89b42688 | sha256-f9fd2ac9e3e79afc24a76060ff76c2fbbe7f470e1b8a65cf67f3718da17b88f4 |
