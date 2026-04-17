# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d734aa4f8ff91d619ea2bd69d87aefdd3f36d0cec38d3997b6f1c5ab56a102cd`
- fixture hash: `sha256-b5a8d59003130cacb6d12d20cb7f35591a0ecaec31de33844db54aba06f55180`
- score hash: `sha256-7c69680d9919e44ea14468c4bad10d5f1fd50d980064431e9c61f2520a493064`
- bundle hash: `sha256-6cafa8d5c6c1171909c8199b23b464c3a2895e4ff12ea36d32f90349ed7e5249`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c822d5a812142cdf3ff00336272092554327fa9d0fe665c2253ac281723c371 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-745e6d381ee00d452340c7b988fdef40c603d4e468a7c252bbce04ee547c7024 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d4db891025888fd9f57080afa493514e53e8b366befc18a5eba92eaab4899099 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3be3af7af124383316378a81e399dcc3f19821ce1cd0f9fd9cf6e6cfd0bf4d38 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-71706a27 | sha256-538c77535c9625f08dc4a6f8636e1fd145eb7883f6870df2f822469bcccf3353 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-71706a27 | sha256-2e1a8a3d1ac7623a9a30b23e20cb8b545efdb6bdc4a8e971905de8c7be4d41d1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-78421dec | sha256-1003521bcfa9aadd48d16628515d436d7b02ad39dbe0728e86958ce33df2f516 |
