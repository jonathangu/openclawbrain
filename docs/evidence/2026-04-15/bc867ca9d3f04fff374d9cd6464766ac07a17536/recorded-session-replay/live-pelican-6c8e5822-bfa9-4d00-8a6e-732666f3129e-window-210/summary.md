# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9de933020cc2b0d03a4b1b4f5bf51c7bca0c4ae8de78af7a2cf6d4b86ac284d4`
- fixture hash: `sha256-2e5835aa933cf2df6faf2714837c2953d1866a5094413604d0ec3e648b5257c4`
- score hash: `sha256-fa3f8486254715431945a3edb984d8145f344ca45b750ff0556300803daabf17`
- bundle hash: `sha256-9a958c4f65d0e4de853111d092e53e05f7f75a519a8120ad4108f40684675745`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13242c9a12ffb4d788c2f14891b978d17c5b819a44b8fb4dd405e1c1b50322e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-897cd83af233f6a13520bde3fbd090af0df04ddc0b986b5184945d3c08187131 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7d83b862bc55981462ee59c3112c3f5611c391ee7333412420de9b5efb7a6dab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6dd62507fbb91623a6b5929608ff8860d7254de0f0dcaf7895d5d51074419522 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0a03437e | sha256-fbff9624851844233746d5066ec29b9e3595df596fe327ae1a79818431c1f506 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0a03437e | sha256-af9c08507b0e4261a69ed201849c87f95bba3e97ee8a355dee166f697d90ec9c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0a03437e | sha256-fbff9624851844233746d5066ec29b9e3595df596fe327ae1a79818431c1f506 |
