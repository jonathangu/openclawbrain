# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069`
- winner mode: `graph_prior_only`
- trace hash: `sha256-836584be6983eaf5fa6eb8781cac34bfbc9fd538fa4b161ac2d1263fee14146c`
- fixture hash: `sha256-d2752f3a765e793797ebfa0ab38ae1044dcc8b2c28b548d73dcfade2be50b251`
- score hash: `sha256-f37f0b6269e6f8a636166fe177ab150b9c9da5670c348ad954cd804a0eb66868`
- bundle hash: `sha256-55bd21567e7d6347338723563e70662a5b893ee2bce8ae1f3b4c9bc6b7016826`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c391bbc2363181b72ba8549d9009dc5fb197cd45a7341e37f0fa91e51803c6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-902407000fdf885775631aed8faa020d59b13b826bc8dde8d5444940ff88e713 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7181de00ff1efc187e2412f9d315b1071cf37474905c9bdaf9b8c780701565ba |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-191503b2bddf89685012fe08ed7b2103296e6f5ebc4efba5f121d00223d76753 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-889cd0f5 | sha256-e554532408b2e603d5870965c9c8b9bf2f415058ef40e2b6419556cda48dddb6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-889cd0f5 | sha256-593fd2e750b4568e5ea7c49582bd527356a5fd866a5e7fe401898ad27a0fd124 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-889cd0f5 | sha256-e554532408b2e603d5870965c9c8b9bf2f415058ef40e2b6419556cda48dddb6 |
