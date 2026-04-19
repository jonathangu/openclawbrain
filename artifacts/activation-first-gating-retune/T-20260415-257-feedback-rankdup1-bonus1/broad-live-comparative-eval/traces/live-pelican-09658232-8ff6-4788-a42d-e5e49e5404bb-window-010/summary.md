# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd69cb2bd54df203880c8fab0fa4c855341f06ecfb9d6ec642144558419aa71a`
- fixture hash: `sha256-f29630fbd2f41b8d395fae06865eb7778e00433b1298788381332e0703a42702`
- score hash: `sha256-938c094901743b04a6766cf6c8ee037a073de0e8fd15d456109be0e296b83203`
- bundle hash: `sha256-6bd1fa7990a3525b86353c2ec40ef61fd6316263f06b8c410bd3d0d91d8e2f0c`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80e65d944c503b7cf482a4ac157c70bd9810fcdc3cd3dc77c36042f87f3356ea |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-9a256cdc59c3d924c26c806463a6eab9e61f1259b9fabdc25f5b6fef7cb95d20 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-5c23014883ad83b69ef2a69d17f8334da4631342617eb76661017b789149f30c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c47e2423f5cdaf680c2b7c0257210439cbd19ae9f3146c3dd0ba0ef94539e1e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2b8b305f | sha256-5846f28fc2c0524200a000bd8b979d7d113f96347e10c1a15d926b2a7ee16fa9 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2b8b305f | sha256-cb12a7b5627992f2c7ab61d3019e990905d78adb558ef4b809e26b9013872151 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2b8b305f | sha256-5846f28fc2c0524200a000bd8b979d7d113f96347e10c1a15d926b2a7ee16fa9 |
