# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d380c2fd8773059a5893ffff2d380e86ca0f972a4140732a5832ea7865e5c2a`
- fixture hash: `sha256-55f386f545922fe7856a581e64b7fa651b1de1ce7956a55af05d3b2bdc86946b`
- score hash: `sha256-d958d4651249baec5bbdf04e73e8da4d36ce490070dd23e069ecf9c766913cae`
- bundle hash: `sha256-3fd032a07a1428ec7a9ec3b293e73264579f07222b7f427cff4c3cd2727c0886`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63284d4d64db3291399ac8e17a28a524a22240af2c68e8497ef443766b42c4ca |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd7c93de3e8a73b64a33612e6e02398f7131181b20c6c6ed82f044a895b21192 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-810db642f917cea2ea0aede789de90e92af5e72ea28de8f6e9b1a4d78825b123 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4953e99b1f64ac939fc911ebf2dc75610af5383eae9aeb46024f7a00f6186dd4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b3c19c7e | sha256-46d40cabf3b03d2a5302c197abb67deddd790b5094139a8a9c164c21c29503e3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b3c19c7e | sha256-babff4678d2d7510b18c9e5a46e81408e788fe0491d268fb14ed9caf8d081381 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cfd4f6f9 | sha256-77ba212ac39d0c87d96a41009b405fc8f591bb8a6000b0bf93bb1f674ef3510f |
