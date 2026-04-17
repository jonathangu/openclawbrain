# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6585559519673a3db32bffae40afa6ee2742e112449b33d0923762a2de179a50`
- fixture hash: `sha256-e5d86f08bdcfbae469f4662e91d9d271a78451f428781b94a6703c49ec68efae`
- score hash: `sha256-e6168c244180ee65dbe86412f627619b81204450d3351ae7eb4d9567a4889b9d`
- bundle hash: `sha256-f5a40c92b46162867a6c722c39429f1ebdcefd37af9727eba9a0938756299d9c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ad1968bf09244474e138b2781f33e5606f4a1c015708e46d2c74447f4594a893 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-219e806fb71cf921c2c000aee9840397088a8963eda9fe6852eec7d79096bf63 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a61445b86bdf0e19c38d7603013518cdd7f5dcf92eae6dee3eb8e6333a4f029b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8596cf1437c53eb322de07ca80ae741e55b878c2712fed95c2e71934664d600e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08339fcf | sha256-cc67eea496435322ad89feb85492b4281d9539ea9dd1acb9b374bc868196a1f5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08339fcf | sha256-d4f1624c2c5f57c08c137fbed49007210f77b885aba8771c0876b08815ac8d5a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-74671652 | sha256-7dfad6906f6f16a49a2c4e8e22ae8e7293ac1727aebea5eb480c6ba2c5ee3449 |
