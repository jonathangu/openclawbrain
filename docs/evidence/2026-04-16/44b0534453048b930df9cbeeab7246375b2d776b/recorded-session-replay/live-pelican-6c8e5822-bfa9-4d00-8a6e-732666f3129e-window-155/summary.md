# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54c6a7f75aa98b64fa06de64444db8f288aa41bfaf9731cc070d54f577be960a`
- fixture hash: `sha256-899705aeb2321d03b6a0aee78d7cfb19ca0d976080db3e6a3f83db60267852fd`
- score hash: `sha256-27804a270e16be6c121eb29d95fb3d919d84167941ba58288d937bdc26fe52eb`
- bundle hash: `sha256-53e2c48b870d42706cf4ac47eaba6de66d74cdd8e1632fa4ffcc07627e53bab7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee198590c5a4a8c84e2f8fe36017d040fb15fc92428b4d0396417de634b42329 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9269df8aa84777284faf96a0adedda2b994fd35794624053247a72642890471c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b7df8f2b03bda6ec3a24b1d746332dbb276f2d51e9645249214471e0717a296 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-49bd106ba2fbc654c2b4fdcc058a4f08ddda43e36f39286d744d2d62a9a6882d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78780a9b | sha256-fde65ff064f475c6998fd0daa7173c3b071c509f66c495054ee432fb7d36533a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78780a9b | sha256-3f204fb97cfc1eb5f38b75fee0b6af15c590dca3b1258b6cac8037282970fa86 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fe0e4500 | sha256-73d9ee385e4facbf7df8b6519f2b4d1eb381af74ada4046f09476e28a92f6268 |
