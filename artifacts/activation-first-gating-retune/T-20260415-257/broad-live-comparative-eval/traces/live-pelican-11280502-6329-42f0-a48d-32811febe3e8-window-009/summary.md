# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f6c0119a771d3d4002ad2796a568648f1ef4a576c6d777928a579e3ee482d2c`
- fixture hash: `sha256-bc5d36362845b850e526b2c0c66165097088057015558a1287a8837f47ec0645`
- score hash: `sha256-015ed456379247d344c065f80c3ddb94ea46a8a85fb23777dad60f4dd2fa88d2`
- bundle hash: `sha256-f5697e01d094274aaad757ae96b8a730dcf989c687c37f12701acb34e5defd9b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23fd42a717917abc9a657ded0642e878a093677ccd38ec86a1e6fb99c57037fe |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-80223d37b2996bd8adb1de12037b4fcecf30a990f688bcbc582b1bc96c6a4527 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-768fdcfff38fc902f1037729945884d32d78afb15358011bf10b2d0dca22390e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-edf30b7701a5ab490f4a6fe30a70f666f7610e1dbeb39c4fe4c3aebd6b5ec089 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-373cc85c | sha256-29a8a27c31d156f5408ab87898af6f513256e3659bed73b95fb20022d6005cd9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-373cc85c | sha256-53d6ae7a9acadbad99fc248581a5abff1ef858b49dda9e7407712c4741de8089 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ca4e78bb | sha256-ae95a77d1e16a48675913c5992e02c39b06bd1c7079c6f5a9996e77643c54111 |
