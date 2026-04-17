# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-428224300aac4f3f06119d7ecdc386e98266d57e94ee4a94bb3cd7cf1ec92fdf`
- fixture hash: `sha256-48b4c185a62e99db11ed2f294b118b4c1031f6e03346711368215f8f92e0f14c`
- score hash: `sha256-e2e07f06af21ec2f44fc726dcf06ee2b0e34cf8df1b7bcb71a785ac259906683`
- bundle hash: `sha256-9a5995e474f2b907cd6f9a007b328762270d242798a56e93d4d9d240dab8d8b2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d4933e52342083e2c61aa4dd43d6b584304761341a273e9d037973938748bea |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f2f6859c7db4fa10a3134c7be46b8692e29fb9928ea29b9ff8e711e572b5477c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8361a9fd07eadc82485688e5c16496a044b44cc903de3a79ea7301409a18076 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ce7b8eef468dc815cfd804a8b57eacb054bb25651a041d3a469055523dc02c4a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b77aa315 | sha256-0f3455952fb6ae3b96121ac508deb197d1bf08229f8170b8bc2e9249b94d736f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b77aa315 | sha256-e8e82d0db405756ec5e5689aa110f9ca379056e78613fea1ee77f0f3d54e3b63 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cecc2a24 | sha256-938ed347aa80fb22607c435983df0963742bd66bc1038dad39bb0401ba5f57c7 |
