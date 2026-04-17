# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc837ac64ce4a5cb1d121e2bea7830254f5b1cd1faf9dd8be0505cf94fe18342`
- fixture hash: `sha256-555eb18092c7a3b48bf36359187522f84e12b063bd73ce65d859cb8f468c2af9`
- score hash: `sha256-2e7ffb7b550aa31427f98484226ff765c9e90f58b37920ed15f67a0c6405a482`
- bundle hash: `sha256-fce7435a118ae99340e0a2996c0129844a6814e5d8b22c99789fc740938567fc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94018db213a88670c23984311d9a8431beabced6aba3b25434ee10a70b79887e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05a32f1c821badd5b85fe1c240957c45d2245287558b2d562860ce74ea2f5f6c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-307b48936b44eda369aa0bba2fa1ca08d40759f53b97965881214e8c5956ceb5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-257efaea47943f4a020eaa227dd234d73939533562fde6bb7b91a63733051d08 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-84d48bbe | sha256-21cafda3a41371d2410faedde30e0c8f909a2c712f644d45b7f402c78e23352a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-84d48bbe | sha256-380dab0f2c803c1a737470c2f9f8b3807e6f0f85ca9479a12e0ed88c6936f311 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-21871163 | sha256-a76dd56be358a077bc3ad81431e62a20bb8e90df9915f613cf711fde6c8311c3 |
