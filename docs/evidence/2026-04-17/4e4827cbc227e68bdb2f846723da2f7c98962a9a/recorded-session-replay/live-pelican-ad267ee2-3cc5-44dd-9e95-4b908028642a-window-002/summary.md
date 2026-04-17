# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1a576ab7fc82836d62896c5506ba892a7997f6c29eafb6387885075368088d2b`
- fixture hash: `sha256-e830bab1e1b5c601ab706b387c4f671be86f28c4ff56747b0f78265a86556170`
- score hash: `sha256-346a20e5a0fd3c75e8b99762741ec3eb01fc568dd1ed1b9d9560bd16731bd169`
- bundle hash: `sha256-ac48ee18db52d818708f8910fdc88bc35d08eec43a5c5536966a1bf76aa3ebd1`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-309a76d5f65b7ffefd710af5c6f62a81606516631b55e10f450624750cad9788 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-af1692f45b3bb0178a95e68a3ca854905ea4d41c8d0eb46879d12295a8da515f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f76b347ca63db4b534634d132b9e7bc068dc520afb4cbbb539caef4e71406377 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6e4c7d2ba008ea3f2fc3db8785f4bdb6961a5d55ecf438fdc8299fe2d03868b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-df7d9d68 | sha256-1282de9c4a8a806a485848a4aedcd06e00d992cabb8d0fed3e846eeeb596a485 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-df7d9d68 | sha256-fc8ed099f9bdf5a72694b3110193ea95291c45bc669cd9d48e4ffc108f866a07 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-6c3f00c9 | sha256-829ed983a164b0c9006c07e201c281b510d79b051493d4dc91aeb0d30679e61c |
