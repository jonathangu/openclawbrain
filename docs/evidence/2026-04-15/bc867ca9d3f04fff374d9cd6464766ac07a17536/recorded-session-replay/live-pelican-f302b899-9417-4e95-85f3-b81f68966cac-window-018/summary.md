# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-367ba0e9c1765adfcb55faa49a77e3f08a37eaf77c4964ca4eb0f5d706e75deb`
- fixture hash: `sha256-c755dcbf454eec2e6cb44da638da71dca0e7b64e802782c096094c2870f2abfe`
- score hash: `sha256-1375abe790e32ab0de442aebdfafd6f13435c89033235d979e285697049a8262`
- bundle hash: `sha256-4ae58862b1932af6017bb592df63a6ce858fdd9cde1785159684b31308e4d41d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df30ae86da7dcb946f187b86df35238c1caa6176c275bd81d1099e4de3972842 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-da04a8362943850564935cb0e05691c07b3045f926b85c296f4627749aacf1cd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2de0efd87d3aa2bd2b6abeb1047e7518b53461b73c09a0888386e6b104d29265 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-53bb6174da3942ca52f0846414a41e180153fe427bf70b7018bb07c6cb5002da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2045075 | sha256-d611ddd0ce8fc27fd8221d471102073ac3acc65f223e53af652fe5c9465388bc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2045075 | sha256-adf1d35e1e88685b7018219530f327656a837b1fb908e4ac9cc441fb361de0cc |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c2045075 | sha256-d611ddd0ce8fc27fd8221d471102073ac3acc65f223e53af652fe5c9465388bc |
