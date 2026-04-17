# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fa0a3e5a2be78a517ccfe2e1e4b8f4e2529d6e5ac6ae4838bd2c1da5073ae788`
- fixture hash: `sha256-9c907c31d6df545ad3189fb66d2746fb0938842a92e6704858a51c0bdbc6d6a3`
- score hash: `sha256-5deb9fdb92342c1ce9ebc66cbd2f7247fde086a7510ff14bd9681c0bdcfddacf`
- bundle hash: `sha256-487a6d699c2efb8f739aabf1582cb9eaaad62e818b103f29aeaebc7fa55dc754`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-984a47e062d84a4c2db4727f0e783a355cfd91d65c98b2c3a27a24fa9103cec7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e56e094e6a9160e366d9f04030dadfd5f4c0f6e9c242b6506ed4e6784c4a785c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b77f2e542d11ff2f5515f03bfaf8883cd20a142c4d56795d048f976dd398e12 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6b329baeb8a88ad5bd64a62f2ef0b079f7d5120833654c348414c1fa7b158150 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec5b8f53 | sha256-3bd43712aa8a52e02fbb8df8aa08ad4e8f30b64f28e71a5543906b2caff7f364 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec5b8f53 | sha256-eb01b5783a8d6bdbe18079fb9d4e5711b926c179ee3608fcf2c3c0080dd4bdc8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-df19ccc0 | sha256-e584046ed7360d2f97c3a1deff2849ee3d1090121d95cf20d12ae49141c44b63 |
