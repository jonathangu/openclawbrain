# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176`
- winner mode: `graph_prior_only`
- trace hash: `sha256-480f373d763bedd2f5766cb9a1a8860701112223bd910911c4830c2fc4277912`
- fixture hash: `sha256-2055d04a6856d7cf43d112e858be3651b8402ee14faf73331e6f59144245384e`
- score hash: `sha256-8f5e819463975246e53599a39a5f13767e2f1206374b425d5b416cd1faff5232`
- bundle hash: `sha256-3022626fec4cf4d402f501406bddc6b98ac2196ca88ef21c7b0e84fb42e49727`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cc6a8a76ddb7a25937feec38e19ee175087db4867c24970d2759b39f1c9b4bd1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fe4934ad82428111399eed91d0b3a45e1335faa72a05e1a8f3f091c514c41b4d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08f64c0c4e8a1023b1a0e4e0fee8acef3f7696dc36d0d8660810c2e7aac44729 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-9852747bd4155a0e6d7f8b65bfb476b439a405903ce3d93d1d2a87f7db00d712 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ea47486a | sha256-e7c8b2818595731271d561a1edf9fdb0798136af5132f10a72b36ed28eb16bf6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ea47486a | sha256-fd9f0ec0402689bcc1505869aa55a32f89756ae7e58d8998be527fa12d93795e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-ea47486a | sha256-e7c8b2818595731271d561a1edf9fdb0798136af5132f10a72b36ed28eb16bf6 |
