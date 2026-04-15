# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c1b296177f077b6c8091fca65eb450be8d5f631873f87466a2fe9011d8b7c085`
- fixture hash: `sha256-56e21e2f0877b996d5170fefdce01e8f6c2815e782b17ac6f82fa56c1dd0500c`
- score hash: `sha256-c940b9c28f46e115792084119f6d26e045900fd865ec9ba48921251b3bc83f6f`
- bundle hash: `sha256-52cc5214e627dba86114bd00f0814fad4495dcac3f9f8d2631c2f55b6d03aa41`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29c5109446f744c540ec8fb2d0eb8a2d5f87ccaaa85851914bbf19fee8f8ade5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-04dd2546adc520db3c6a61a77d2cf1bf11ed4f4c9a8b21b73ba528fa165fb665 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dbf661ea85d386ac76b138df4c11a7835030298d397f4efd2d6def3fe8b4560f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2dbe2a262b5d520fc51946f66f0a845a98af0aa213b3be6cfe649afba83d5886 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-778b4c88 | sha256-2406c6683d539d867181d0e67ff1afec3184eecd52f06e896f291c879612e29e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-778b4c88 | sha256-70a238d8488ad91a770ba4a9e3df0efeea8bf632d252c0c4d82a242164585da2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-778b4c88 | sha256-2406c6683d539d867181d0e67ff1afec3184eecd52f06e896f291c879612e29e |
