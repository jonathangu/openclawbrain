# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-172`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c110d52fa9d814d5415fbbb31a6466d9c241d27a71e256584d4f8da38b7870b3`
- fixture hash: `sha256-7aea2d0a1eb139bffa0a7ec4a62af3e2b3a4882d28c7223058abf7f69edb1954`
- score hash: `sha256-81cb38be57af8083d1b8178c34291d47acc2044c3e6b76c51641312436ae6e74`
- bundle hash: `sha256-554198cd2281102fd855552c3e780f82bb9eddfd16856d9c404a28e94849440b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3c530b06d54c1f577e737bdbbc2ed643a0051ad5929f4a0256034473b3d96cb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88bc9cbda57d4a0a264b55172b64d9e9e8d88e87e2c38efce176a6708678832b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f814212b901443438bd324c469c6bd7f23d5d4a31459e8c557200fdab154bad8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3801c9842547b45076a8452a0ab6e49a508fd84827747223708ca461330b4276 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9dc589a2 | sha256-2cee470a438ce26c403d3f95b36e07ca937b2cc4beb7ed2423c1c5b6ecc99ffb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9dc589a2 | sha256-da79f02576430c86029285adb7dc12ab7073c76936ee743bc2c6e086b063d6cb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9dc589a2 | sha256-2cee470a438ce26c403d3f95b36e07ca937b2cc4beb7ed2423c1c5b6ecc99ffb |
