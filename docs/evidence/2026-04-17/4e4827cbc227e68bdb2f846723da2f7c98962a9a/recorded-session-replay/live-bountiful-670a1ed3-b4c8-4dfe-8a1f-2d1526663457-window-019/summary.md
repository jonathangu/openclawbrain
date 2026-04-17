# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d68f21a3db07e083abd55cb8f30309dffa35aea63874e95510f19d0d69cb1ce`
- fixture hash: `sha256-370af296b8752ce6655fe59921b05e957209333f8adae37b056699cf10a9af35`
- score hash: `sha256-4d37e5cde8aa6989293377ee0141fb0a572d8e5e7c059ec83884576e98a4a4a2`
- bundle hash: `sha256-712a5016b29f3141737490a548452216b48bf76592af8ac6369c64fad325d0be`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13ac363ce7285d3640914d39071894fd6c80687f14f6807f8531ccb47249088 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b77ff95a23c989a626d5e996e018b7b926ccb0f6a1d15540623e5a3859425c8c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-19f2bbd8abadcb778c513b54ba689d5161c4361c5fa7da6ae779610c5a7637c0 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b00997b6789e56d44ddf279f6aecf339a82abc7d99e4004790dd69e995a3a6f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4ac029ab | sha256-eaa3ef5a3bd151009cfbc23b3d28cbdde05e3f7f0edf37b325d4a123dcbb97c3 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4ac029ab | sha256-ad7cfe22cd0ee7f9c6c9359e6a638a32743b8eb07a50bc92dfc307f9a0039f76 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-58ddf136 | sha256-8709b2002edeb931f693f86e2b1da46cfe0584df03718fc56d9ccf0e0f1ffa82 |
