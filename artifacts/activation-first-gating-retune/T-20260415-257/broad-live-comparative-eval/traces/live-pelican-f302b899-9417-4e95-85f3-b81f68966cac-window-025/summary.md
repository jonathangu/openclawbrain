# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-025`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1e91f891be11ad983e343a9bbb8eb7e094a3203fdeb0cba32d80844dcceadc5b`
- fixture hash: `sha256-c962d7bf59f91132e81f529b35b43a46128d3cc144f19a803783e383eb2588e0`
- score hash: `sha256-3010d091f148aefccf3eed94664dc9c1b5617a765db198de6c6a4e10cb4dedcf`
- bundle hash: `sha256-e9fd8fbb634af85ba776260a4f8c505384c5c4e5073ab6ab86b10581e3bb1c4e`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5abc89ba1c4aafac24d8b492241ea58c50f7925494e6166e3016c9a753e61584 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-65ceef5f99e6b219974c127762d641943a0df580b1c0d6251042f94ce358a6de |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-219e6f5807a0d3830120d64aa3a25ae09e2b46e659c51b6077224dc9181c0b8f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-31cc32536d2cafccce521d5cdd71dab7b21a20d16b7bf80103b384bdbdbdf5d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e50f3e22 | sha256-117a63931231669e449ebeb74c11d70dac86037f3fbbedc2b05e2652550a03b5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e50f3e22 | sha256-0aa9d1962aa9044c6cc5eb677da88e9cc8276691f1916c054f6d7054795989e6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-e50f3e22 | sha256-117a63931231669e449ebeb74c11d70dac86037f3fbbedc2b05e2652550a03b5 |
