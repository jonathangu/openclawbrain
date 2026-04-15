# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304f24fec2ef73f307ee3b3bfeff3d4bd90894e7d9fa693794ad2f916befa2ce`
- fixture hash: `sha256-0d0f5f0a3dfc50799aa0a0583bb1e17204f3f01f50323b91030ad8276022d234`
- score hash: `sha256-5ad80e7c8d82ec31ffd836f43e2cd2f7086895f63a3d94cafcf78b8b395c1a5e`
- bundle hash: `sha256-ed84f1b1a2e1c1581a4ea593b4de540175177e88c3c8ae4f5651af920b0af0f5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2db02be6aed4a54b1a82eb4486629fc6a8c812b69fc8bb1feef7e61858ba9b3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b05e893df28eb6028f7e6ba04968b30d82700855f037f4915fc4d272a059c4b2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6c60c40f003ee9f03dd91f2f23348c636a87f57e101086f6e42154360aaac9a5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b5ffb0132ad7ea38f791568e31574d4b520adc33f44005a20648c2a746208b70 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6333d243 | sha256-b59a177ad6fd0796df405e4cd6719f3864e0573234beb00fbad110eb42469e5a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6333d243 | sha256-548803102c5e5fabc1e131ef90ae64f2c62dedc2fec841bf9ae4023f43eebde9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6333d243 | sha256-b59a177ad6fd0796df405e4cd6719f3864e0573234beb00fbad110eb42469e5a |
