# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-419270abab63416f7d92b578d4fe979350ddb0300a46b65c46c842d826635713`
- bundle hash: `sha256-315095995a4a6261ad1d0e4783a704e493be68b300acf3253819b341c9b62931`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eafd155fdfa2fbd8e1c5739855382bb4aee55ae760f037b37bc2cd66c8f2b4cb |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-46371d3f9c1ae4c771abe94267d3ab2dd0b34791b332cb09278d22e07825c54b |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-5ab1a1aa2d0391af667462d8bd484c8716c8906947ac45239a5822be04a4c728 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-62c3eff7c7f855bee03c7745e40d9f5600b8c92791347ec63989105ff32356fd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3f50b780 | sha256-a7795a1d784c0ace5d740479364e1ffa93591f19464ede94b9c09b8bb80d627b |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3f50b780 | sha256-f1b47f5e33ec896b9e30f166397778be02b140c0d816e0afbbc4545351aba56e |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-464c107b | sha256-6434eeced960c60f1b73d6db68ff5d042546ff6afd8c2452ef96310680fb7f12 |
