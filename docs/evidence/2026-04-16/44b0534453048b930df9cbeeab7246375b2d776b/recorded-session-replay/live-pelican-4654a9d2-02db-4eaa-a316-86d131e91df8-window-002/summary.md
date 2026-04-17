# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0107560e9fd434b7938c996a94e09516e9330df1381928365035d337054775c9`
- fixture hash: `sha256-ccf24038ed94c209310a49ea52fc2105449214d461e66d2dc1493bec54050346`
- score hash: `sha256-d609d2c40bed190f03b704d4d093548d38424cb5ccc5005f1efb1725d551ab69`
- bundle hash: `sha256-db2f795036e95b1914eba5698ea515959f40a6401d900c167df71f479623c3ba`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cd39ad77bdbc78ea0e62a163e0f69b70fc53f35c07ff18076ffb99dd86c22c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-213446dac76c1d686a2fe7847d2e175401021b59bb08ac985b269a3108d7b498 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca23c079f51dd01c09fcc15f87032762a8eaae93dfa0f271359474db660a517d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-79411f62ae355b26a2b2ca453956b8d04f7913d7bc48eb3f624f571762a7ce2a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-da903f51 | sha256-c653c80dd21df3946459ea0a3402559d63848bb9b8bc4df50645e2b3fa5e9648 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-da903f51 | sha256-24787f47a0d54b7a8d668227b35372257fb7f37c884279f0889bcdde37653e17 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1b4e6ce4 | sha256-f22199a9d9ca792b46412322a90e62c6d058ab43a46e5781f7981618163baf7c |
