# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d734aa4f8ff91d619ea2bd69d87aefdd3f36d0cec38d3997b6f1c5ab56a102cd`
- fixture hash: `sha256-b5a8d59003130cacb6d12d20cb7f35591a0ecaec31de33844db54aba06f55180`
- score hash: `sha256-9240bcce4c1b24a786f1734044014ff413789831a387c08681035efb7dadd3a6`
- bundle hash: `sha256-316e1dfb953c6f2dc068b3b9134686bf14fd8cfdbbd0a60826ab8a40701dc833`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a064cd20b0d9f8a1ca8ce3610be473234489376da5922f1a553f8dc673e1a9a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cd0bfed0c079ce26967e895e01f72361c11bf039e58521561ea13f26f5a66d35 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-66c75ac0c8abd23c69fd5d50b6e1ee354628e759eb3e810abb959b19a32c43cf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-30ab402c | sha256-dc259b2218a2f25ba6cc642abeec50a4198f4961132c60528796db35cb5c0db4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-30ab402c | sha256-4235f69cbeda59394ccfe38c417925c89e9b32d5f8c5ff50c2edeb52f7d5e351 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-30ab402c | sha256-dc259b2218a2f25ba6cc642abeec50a4198f4961132c60528796db35cb5c0db4 |
