# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9abdc2f8435606514daaaad4927f60e901a9d2b092eb5d39df77887ebe5a304`
- fixture hash: `sha256-6e857ab9cb3ba1ec3e0f72cceabb24485f23daf6db41d61af726b2888aeb0f66`
- score hash: `sha256-d7637a39e902b17d5692e8f49e0b64063d656546ca41a5bd5d3dc230639a4e8c`
- bundle hash: `sha256-a0f2323288897e3fdfff7ac64bf6e87877bd9b5b0466b3cd5079099378882080`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-865378fe979515e6fb05b86bb93e571f4e3d4c4ed17ab843485b9830a42b2636 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-98cdf39e1abdad69b01d419316cd3340dab3e14a09739fefc99203df7330c776 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-84f25d0b32d53c2eeb895ba9a51ff156a390f888657259468a914bdc0b8c86c5 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6b0555aa436a8e440afb58c3dde45f2189eb372b1bd4c843704541603f26a3f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cad7fbde | sha256-59774b9272c7ae6a2de8e720bfea8d41f3ba658539adc50d7022ddcedc5765e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cad7fbde | sha256-a1eb56b47993cabc26f2c9e4217876a6ef9d1cebe4327542025aebde19d80286 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-5b11114d | sha256-034333f351602efd7496c9dbedf2772c060812ac11440d21e5b9a0402d3c0aa7 |
