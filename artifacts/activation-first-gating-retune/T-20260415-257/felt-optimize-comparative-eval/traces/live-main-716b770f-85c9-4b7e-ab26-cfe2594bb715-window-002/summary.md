# Recorded Session Replay Proof Bundle

- trace id: `live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e321442dc8033dd76db95133894d776ec05ebee5a5a98eec612f6b420b907658`
- fixture hash: `sha256-742118fbdeeb061b08c45664c524844d158f1b6be0af589fa277c4ab60f660e2`
- score hash: `sha256-b3b769fcffcab6398eb851ae0fb5f52f718082afe96d783a3644f881c29ee4b3`
- bundle hash: `sha256-d67dd35a9bd02552f1f3c1f353ec6bf69635d12f294a00e4088b5b84bfef3dbd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-05b0912208f70d1fd8d2baa8f914bf08175b3f38b8f85e68cab4f50d835557ec |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7fd6d0d9bb1fd372aecb1580883e755fdf44fd27a78922f96eb4292317d3e949 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1976bb97cd37a117bf3198f2d115f21abf907ea6c32b6645bc0d6aaa64fc0657 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-af5b6936c5d0ba482c3fc0b69ab850c2f3ae43537a60172a369308c6fc6075aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ecfb6ca1 | sha256-e7f17366e73f5f824d386cd7f90d2d3bec389cc69862e5de51122a744881c10e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ecfb6ca1 | sha256-c75b1cf2160fcadc4f3b4fe4c6dc75298d09d349d4eb11c0b287ece778b5649e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-dc3dbb36 | sha256-69175f619b1f854c81e452e1d99d3e99faf689efb47994c711a830b7e26fff1e |
