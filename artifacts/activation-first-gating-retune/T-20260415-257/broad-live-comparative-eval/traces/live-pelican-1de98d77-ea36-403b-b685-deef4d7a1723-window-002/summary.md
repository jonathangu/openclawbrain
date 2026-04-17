# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b39ef4fc4945a82dff034380c9080960d0e6ed5fe56fe5b4657351529db21cd7`
- fixture hash: `sha256-a795947af952aa839da230500896d2e52bf78e338ce72dd740b6a925befadf59`
- score hash: `sha256-42f8347dec137be982a82592b31ff15d2c0a2fc582d0d893dd7795c7b9c78997`
- bundle hash: `sha256-23b3bfb781dad870c6a55a70b4d55c5b12180fb0a642d4a01bd5d8b89b6c2263`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64e7031bab11acf7ca7c6563e45ebf707e8feb9b8d59eced338f7e5e56bc854a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3d9b5398a3e51215ce8538523dda2d823c7dcaa8007a46d26261cc20ff9dbb5b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-42051b81116f7bbd6992b715e1a64964343e0f713df3d19c3d932fd661afdc0c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ec1077b9ff1d37de8bb81b4d653cec3992da359cf3553179a32727877f5fbd31 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-710b2d7a | sha256-7a8ad851334ddd28b976f818415f1611ca519c251dfa385d835676f6e3ffc173 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-710b2d7a | sha256-3abeb2a5e0f2268ce59a85fbc518fdda28f23a646240f1d76edfe198fe380258 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4c20858f | sha256-cae9cb94ba96f2eb3beddda35423d9eec71a02a1e40d51eacaf34eabc8bb445e |
