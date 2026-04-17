# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb0ffa9f4e8d70c4fc8ddccf35cb362423daf7116804c6734e40b9d0f4296bf`
- fixture hash: `sha256-810d932d8ac4f8eed98f074f82298ad7f5b0354d5fdf19533c533df6c21240d2`
- score hash: `sha256-8ac6465fdb2686d84c5c6e1ca33304dca7064322c03c729fc76464c457242028`
- bundle hash: `sha256-434d5421a067e717850fe7a060491031414dea96a9868f63a97a089a65fdc43a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe2cee15d245f859cb5315bfc802316abb7874a3bff97839e84f3440b5d4a896 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9df8c0de829d3aae8e7141919c46effd0bed98e6babf9b3ff1d754235e872124 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc0c3927fd5e0c40309d44d19c57a7beb0535439295820f73918d1cde09ffd38 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2d34b404a480d8ebbd2edbeec6e6ad98f9b544142addd714d0ba48a099597d66 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1b1ccadf | sha256-0651b26fa78b678e0667d5a69f981cd88ec521d36154de5359938702563e08c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1b1ccadf | sha256-23173965cf6a33736261c99b27628d42f1525f44323dce6eccf7e7b144faa445 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-500cd42c | sha256-fc689aa1ca0151068f8805f67a23704793f21a3934ef7a962fbf2805d00c2637 |
