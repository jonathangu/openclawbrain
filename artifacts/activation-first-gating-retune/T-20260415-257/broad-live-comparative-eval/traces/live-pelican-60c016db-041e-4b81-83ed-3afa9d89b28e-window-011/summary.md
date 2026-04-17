# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f8ed2037bc2ff0feca7432af422fbb2b58a869f3969fbcd41ad42699329bf723`
- fixture hash: `sha256-b54fc3ca6fe17a912f89c0806fd3df709e1f6f80264d7323adb898abecd00677`
- score hash: `sha256-9f343dfb67c4f6c9ae158c5497efe71127db131706693e32b264068d8e9b1e47`
- bundle hash: `sha256-257b385e185efd0c3b0b29e8f53757e8007719806eb9eba071cdc015e045bb22`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-34738adcfd10c341d46efa61990a3844e1795a8fb18b5ebdc9694342a06a5142 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bcc049ad68f76d4ec92e389e88a95d94ac2044e349320d606514202ce2daa6d9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f538abb0c0ea0176a4bf2e273a3a5ba258229eb043329858c3f6cefa85bf6970 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-bbabc53a9b204139d21bf634c5a89edfc6414b6b115f47a233032d073c34a097 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-80369668 | sha256-361c8e2d553962db0076658b61b7e4b9eabab8604f7f618d49ea01e237ba1f72 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-80369668 | sha256-58f35ee5b32991800399e4c542e325ace2be7949a08b24d49909fc21c0a1e86f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7b25e579 | sha256-df05644c8f19646e870c3003e2df3c337fdc8d68ba7b26191c22de6b9a8b55a9 |
