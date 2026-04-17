# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f65f7ec4c1006917225f6f3df2297434078972719c9016d9a4a28c343601c090`
- fixture hash: `sha256-0846d04b26eef0a1a7c06190a5a1fd4f54e0a1ec3fcf3231ae0df203565132b6`
- score hash: `sha256-c8694a3206fdaa34ebaedda63a58726ede24b029942d4d0307734c073c326406`
- bundle hash: `sha256-d0da583685a523c98dd4153616ab571142ef2f2eaea83a67fb14422971ec5d38`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42f48b48c6c450f0664e256db3a267d908035a318a1c9a74a979a0b9949d1634 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c05c8178ffdfea90eee4c24d64c409495b086f868bc7482659a143722222b17a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d087e89e67373ff01853953b475b91afaebfc88c55f3e4d689a5ecf7a140e8ae |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3cd12366139eb4ddb8ccb1cead72ee3cf2202542cc72e142f0ae6265f0a74df7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68ccd8a9 | sha256-333b8fff3f1a7516140711851b52c5082f9b42492ac20930853e0d7387f72a3d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68ccd8a9 | sha256-9a1ed12a25ba63e4fa9cb9cf50f2fd5ea3af36b04089443a4845b9cb0acf9172 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2121930e | sha256-fafe66685192ee0cf83dbd1f1604e0daf9393779ac75ad3e2e6e0b60932308b2 |
