# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dfa81f4d2b00217c5c5c520178573740e8780c6997e7fbd463fe714331cc7869`
- fixture hash: `sha256-ed00dcfbed6598ace12042db40479b3199c9a2955a7a673a786b8d8fa048ed17`
- score hash: `sha256-c0802bea90ac7a16dfca0a604cfd974ba29b7c018198f35c9bb2fd8194d9bc03`
- bundle hash: `sha256-dab047e7eb22d5250742c32d59862856b7d142ea64996a292d8f91c8583b8792`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d5028651743004fc65c4abb7a18a3ce781f93f13bd67703dbd698c51e61ae2 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d87413ea3d44b0a5492a48834894e7258dd11e05aab27bf41e0e6cb071a6c949 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-c1c0aef0dbd291b29c45fa8c117dfc82360e01bee840e850efecffcd65c2a352 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-df6007e29d0c22db82a2d3389a381bdcb83e2dd25de4749ee38138c041355296 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-ed07282f | sha256-7947b1a49cbd38af0cce7e1ebff8c39832f8cf7fe07393880460832c1e6d414f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-ed07282f | sha256-7947b1a49cbd38af0cce7e1ebff8c39832f8cf7fe07393880460832c1e6d414f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ed07282f | sha256-81fa7c61ba1046239251c1644f99482563af6f1d7e4e05ae24cb3967d916416f |
