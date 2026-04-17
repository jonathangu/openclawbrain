# Recorded Session Replay Proof Bundle

- trace id: `live-main-716b770f-85c9-4b7e-ab26-cfe2594bb715-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e321442dc8033dd76db95133894d776ec05ebee5a5a98eec612f6b420b907658`
- fixture hash: `sha256-742118fbdeeb061b08c45664c524844d158f1b6be0af589fa277c4ab60f660e2`
- score hash: `sha256-aefc21c45a35d3f73601b886edc9b52f42f86de65df0bf83deb92d4ff1432564`
- bundle hash: `sha256-3907aa5f0791748d515f85495b1f984a60581c9df8e4acd64d2077f8ac9d2029`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-121de642d165e3112a3b82d6be7e89c952dc2ef6937621067f7dd86553b35de6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-53b56fc5e0fb148fa6d85723b90665810c94ee8df33b65e022dccd11df83cfa2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f594335eca0fcf943ee7598ade5b4bf0d126c642b2927123185a8315a2ff1122 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e22b0491 | sha256-15b2e31c828411475d66d8cb035a54798692eefe21d71c2f8675d71d0451f9c5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e22b0491 | sha256-633128aa0fd47258846b95ea5cad250d4c9fc8e7e5868587bee98c76f72cbed3 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d16d5326 | sha256-56ccc5f9c9685f6f6301bc01688a3d1e615047b435b6538272b0411ea17e014d |
