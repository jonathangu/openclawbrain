# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8588f62e6cb39b6bbebdb00e938513a4cbaa506b41be87532a11b4304976dd66`
- fixture hash: `sha256-523f979d52f465f7796de01a235f1b7bbea1b624b0a2f4aa71ab4b02e1ae0958`
- score hash: `sha256-0ed1f3981cebc782525c3629ba78c9c396e8587e7252aa1ceaebcb5573235b4f`
- bundle hash: `sha256-03f3b0cfc13bc5606156ed0edd5b347849bd4dc15ce6a6b61f4b852fd06752fa`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-172463d69f5ae184c08f379b77a680b592819857917ad8f3596af66f22037f0d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76585a875381c64a27ff3fc3143b75fb707de391dd5d38050f5307326aad485d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e5058e303f753b1a26422b3b42d58d1ccd6bfc13f8de5f6ad5e29f6ee66282f0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5b84d1f5eba7c12eaa461e8a3fb740de7f397997d01542f162486d648dc87e25 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c07101 | sha256-c6f80cc039af91d4d6feba983aadb40de1980806add0ec2b952e9ed4878cea73 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c07101 | sha256-b66b7799bc11ccdd0c3dc68e8a4dbbd8d776928e20677bf8e1702aa6b74600c9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a3f54d90 | sha256-865616b63d163e09967ff4840b59bf332c25c79ce4c5da32dbbaa484cebee064 |
