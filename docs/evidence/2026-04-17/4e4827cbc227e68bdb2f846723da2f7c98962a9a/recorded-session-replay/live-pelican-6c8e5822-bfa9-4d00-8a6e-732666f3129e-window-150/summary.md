# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5ab4245c094a4283a2fe623f159ada638f3a2335ec988d52906db167e4a412cf`
- fixture hash: `sha256-200539ec6ee07f9053b46fde1430980f62e83407874931f79115b5f9bd8b8337`
- score hash: `sha256-af86fcfba561acf89cc919264dbb79783482b1e57484365681732582b28221b2`
- bundle hash: `sha256-04bc404c3cfcc4ac5f0a8754aadc86a0bde39e8e2caec441065ee0446ad980e5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1ac429de4ad281ade866e710d7bcaf6542300ac52809bbbdfe005490548973 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-edf7bee94ea19c29220a61c08bdc7cd76946d4fef51aa127e99f1012dcbd8d13 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e387e4c01dcfc7b6e6640fb39f28b9e6a2189f07d3e39b63959dda7c20af181 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b112b7c379fbe15fe8039bc802a7e464a1dd988117e917e205de2faea854d912 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8837fb20 | sha256-dbd54f01b028127c2dbff0402f53f7e4296883a1ae7e89a4849d257e400cbf97 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8837fb20 | sha256-eedb17bcf7beb292eb634d7a2660dd222223179ad4ad7cc246ff3e2c04f77046 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a5a17771 | sha256-b4fe9bd49fe070d66809fe615509b2e18d43000b63786a9b691f357b8d322b62 |
