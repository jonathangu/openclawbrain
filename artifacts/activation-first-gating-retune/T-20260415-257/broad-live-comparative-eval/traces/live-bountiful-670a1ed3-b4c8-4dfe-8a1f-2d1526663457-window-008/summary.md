# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f10c00dc4efb180b5273900a9e561d1c614344a77050359aaa2d54aa27cc20d2`
- fixture hash: `sha256-43065829df1e95ca79dff07d99e5773679b5561b6bbdd3945d317201ab2cca51`
- score hash: `sha256-6ff13651ac6cdea74c4dbde287117f11b94cb95f7e7d5335326f11d3e3d1bc7c`
- bundle hash: `sha256-f92b51683ffd685f4a9c9f7bdd0aefe43babfa5f73158ec745282f6e7242a01a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1460fd13a644dccb389d5e4bb97bb20a28fa61d221da193a36a1bd2b7379c0d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fb01bd7129dbc428c4713f960c5655abccda523364ea78cadf2fcc9db89ef851 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cbd54590caa21dec11f31e102be72f7aac46fa360f2bb64d7104b1289c519bae |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-a95c582836fd9f87c95514a2d34dddc5a4930576de7d811151ad302522ee18fe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-daac20da | sha256-c037f1e302b17c4f432563d8b18c8e70b92fbe4c768602a8fd04fa831bf3c164 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-daac20da | sha256-c037f1e302b17c4f432563d8b18c8e70b92fbe4c768602a8fd04fa831bf3c164 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-96ebb28f | sha256-1d6cf1af37dd909678e2237fabd6aff034fad24d36cf097eea909959d02feb03 |
