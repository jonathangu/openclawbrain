# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2efa642ded94688f4afd0eecf5358476119ded4a7cc615d3107a42bb9da56bc6`
- fixture hash: `sha256-8ea6f9e0676b2eee5f6719ce0cfff479ea74f9e0b50aafd2d0b799110ba4611f`
- score hash: `sha256-22018385962673db1674b79f805fefa10a862380af9647416153862e127fed4e`
- bundle hash: `sha256-bbec562d81323e35ab470fea9490e85bb785a3441f432b2295f028607347f6e4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31edefd25aff21f4c99c143ff7e36055bdfeb17ee0aa654ab5907f063e5a4c87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-680fe1c53ccd8b6e7b4ee2032fdb344c3fb7a3baa0393a81e526b53b60d68edb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ca11fcff46442bf096a628fbbeae0d2f2ca82645aa5377776cd7102d412ba106 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-10c17aaf6b8a540a17945f9d4189b3e08ce7049938210f03ee47c8844f10a918 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-331c5ae9 | sha256-74f86cab471010eb0bf2fc382ba7749370314d7bd4cbfbfec428dd4c675100cd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-331c5ae9 | sha256-1b13db5d213397419576c351c075b0be64a64e4d66718d2cd4d254215f1d9504 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-74067fce | sha256-9c4ba46e0360f1bce95b995734fb262c73e411b9698229f62f384d64d59bad2c |
