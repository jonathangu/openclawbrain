# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3e2ca93e26f85fd9a1adc63c2c9bbf2ce46e6fac1e384fb004554e3bdfbb894a`
- fixture hash: `sha256-2c882179454e5a495c5e21a4e1c041932e6e22fd8d004e6866e2895b395e2694`
- score hash: `sha256-260f42be064a6378acaca105888f64042451bdeb990aea369492b7c332cbfe23`
- bundle hash: `sha256-9737eda4c519aa2c92143d8f6bb4f02dfba326a05e876dc78e45f626e4b86b17`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6239dbac6a5fe5a4088922717f6aa5906be06e4cf4f4984972709cb77ebdbdf |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-b868b399044bf92a220680a025d72204855bcaa1c1b2ca4ca78ec7afaf35f6d8 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-76c15adfddea61a2b56279072fa24cd36f77c9dcc8bec37f5557e9b96fa84532 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6d0a6b6e38bf5c0df43d3e2e361a264c88a33037054146cabe99fdf8e1de32e4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-d2dddf68 | sha256-33cd79bb820e1b707df1acc52a684677da818fc1417c3d8c816d83cba0987702 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-d2dddf68 | sha256-ff20934fe555c350e21e20ad08a91d1fefac5656e1738140001c4463b3d9e638 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d2dddf68 | sha256-69018f6d069190e2fe0f6834ed74a13749328816a67e3f8f07e09b145ccf9c4d |
