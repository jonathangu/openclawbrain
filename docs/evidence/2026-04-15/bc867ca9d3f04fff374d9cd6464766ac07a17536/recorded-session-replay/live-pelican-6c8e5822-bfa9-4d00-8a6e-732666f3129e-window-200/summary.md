# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200`
- winner mode: `graph_prior_only`
- trace hash: `sha256-438b689c90e1516f117c130a44f955ebe5121f19131ef3c8af4f3b72e782a392`
- fixture hash: `sha256-fef64d4e61173927de1b8c7e42759f7ee5918ab3e67738573626a046f39d5b5e`
- score hash: `sha256-7e260931e9ed9d192f8df8b0ccfc7ace1254353875dfb4318cb833ec47b7bcfd`
- bundle hash: `sha256-530c9148866407d41e2f7d4e951b6a95b58f686f2f318fb85586b205c6a6a8d0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3e6c6ba4308832d244620436e1eb71e4969051bd02e8a257e4c9a12dea8653e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4be4c8940fbcda4558dc5c4e7256203c88ae2d95081f4e1a9368a3b490ea01a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-21507ca9fbc2de843adf0e1f07ab14c46c7774e4a0bf0d4fe73af64716ab66ed |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-570ce314019f5bc7be4eae54cf3de6a79f1404855bddf1738735f90c58e9200c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-baf80f8b | sha256-7dd6982858e948c38ca3942ffa9634ffc6812e692ca295d2c080e279ccdb2f67 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-baf80f8b | sha256-52df72d3bc0f4e43423c3ffa491876b13f35eeb48f29dc7ea96d55f7e36c12e4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-baf80f8b | sha256-7dd6982858e948c38ca3942ffa9634ffc6812e692ca295d2c080e279ccdb2f67 |
