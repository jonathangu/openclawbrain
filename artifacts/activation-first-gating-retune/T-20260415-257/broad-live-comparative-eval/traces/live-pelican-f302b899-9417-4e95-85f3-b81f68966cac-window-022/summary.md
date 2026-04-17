# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5d4b7ac6ed69712b1588ada5d64482dda6216ae5bbb670a70c4e5011448ae050`
- fixture hash: `sha256-c583a0a30dc7272198329e0ce06b64ff4fe39dce1f96b56a4f82e04f4a924ee7`
- score hash: `sha256-58af0f6814e2ca4129fd8d1f063d53b7811fc22f21c8447d4542a8715809f721`
- bundle hash: `sha256-68b47bb1987170ecb764be58c89318a4d476729579532851ed52de9c38a6b7a7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-646908dce1c2aa715ec563720c445a9dc7233e215511f30956abcb8a6c0f9113 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b83d547b14e5186af9ecf0994a085196fe674fb338c796783bb8c6ece124cd11 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88a30ea3b75b205592ae443f427dafb2928456d97dc3fe0cbddf916bca62c66d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-28321e77e9a18ccf396d1a564fd40dc660006319d0ebd1ffebb0be4ecc3a362e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-444def98 | sha256-40def197794b298c1e3a902d7ffa378ee03c80314d6c60ef08eab803c1b267b2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-444def98 | sha256-b51ece3d07fd262d064c16452e085a46a56fa4c344db49f2e336b49e6afef011 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-31bb121b | sha256-368cce94274dd55abbedfc62d323d46019853034064d4c1ab061608d20a9f912 |
