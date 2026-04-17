# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-05d1bba470fc4f8fb9f82c7e8ed03039aaa6266ed8aa6169582411c0ca7d52b9`
- bundle hash: `sha256-7681a5cc9f01a3b231a522eb5206cdb2b3e1980d0c89625ad95415bb499c174e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-19f888221979998d00fa2be197c23da17970b2ed72065b33311cf29d18943b77 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-94595cca6101925fcf2141a5b4b36d3be029debcd5b9f67f9c5895c76625581c |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-ab59722c81d47431809e8f20c30b64797800e5533ac7535ce83693de05aa049f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-eb766eac | sha256-63a1503c11942f88535fe56712f73221a142f2d36ec31d4d21803948064cf47d |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-eb766eac | sha256-6ff4fbcc87f53e136aab502820cbf856306e3e25e875dcf0feea86a32ef1652e |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-3d2f8f59 | sha256-2fe531dc3263f9e54ac0190fb7594abd92c0ba3f1bf8e65b48a18660c53efe84 |
