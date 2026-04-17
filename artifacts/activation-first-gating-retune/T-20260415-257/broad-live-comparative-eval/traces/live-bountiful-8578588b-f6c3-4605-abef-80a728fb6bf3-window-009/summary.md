# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1fd020946f43a8035e7b5bddd63932b365f2da8b3586893d3d0f370ca217a92d`
- fixture hash: `sha256-9e494f0e3b9885b669e057408d4e8e8d1d86d287d75fdec54a2b7380f98f07fa`
- score hash: `sha256-008d816e264ecf6e741f32ea971cabbf81e54f6fcd4092c594813c8a9edda8e1`
- bundle hash: `sha256-be955f04c8d143f271991c02b46864624de90a63390f72a822ca10542880cc1a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f82956037e9003d716b0eb45a53efb59bb4a7228c918209cdbf92b9a2ea73fc |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a9baf723212da94b401feee7e56222aa253cacfc454b70dc23e054c33e7d80c7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-23b53c80c1ff60b0c8f63bc7c76be5e016adcea91412f19eec0c564c1b9c9038 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-04945771618a2e1e716435cf8b0e0b417191ec2a875a01fc8a328207872cbcba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc5709ac | sha256-dc10efb69144327e2fc6ff7398f2dcb61276654b69835c4d252dd58bdd8f398b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc5709ac | sha256-660af57dabd50f289e5b2c98de01b67c8a3b935d1b4f3dd3e09ca0157babab2f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5c27ba27 | sha256-fce93c5e8943c356193e2fda4663ede1585423746bcea066501d16567ebc177b |
