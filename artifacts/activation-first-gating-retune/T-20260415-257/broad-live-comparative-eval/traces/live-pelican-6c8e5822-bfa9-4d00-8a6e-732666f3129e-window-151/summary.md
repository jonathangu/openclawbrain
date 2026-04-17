# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c1b296177f077b6c8091fca65eb450be8d5f631873f87466a2fe9011d8b7c085`
- fixture hash: `sha256-56e21e2f0877b996d5170fefdce01e8f6c2815e782b17ac6f82fa56c1dd0500c`
- score hash: `sha256-6c300b48fbc0b9897010692d47eb10fd8f0927a69c1835eb5c9c864ca1ee496e`
- bundle hash: `sha256-3985df17ec165f7e7b3a45a3e7fb8e3a9877ebcc0e808f7eb158df73356f5eaf`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a763a189a51ac9f3e34b4d2ad93e0e259c1bebb53fe6b426bcc7c5ffbbbe3e3e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-354759cfa4c6e124b9410e60bd555f03e23b2398f672186d5bee92ad5bf2b8ec |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-566cb1af3ace399ed785322ae23d681d3290e3096b3927c6737119ab0a8220a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7aada54b | sha256-e31d0edf404f676d6d6c9722b0b58832ce59bd078f0b515ee192afe14b0ac61d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7aada54b | sha256-e2632bed1df86ea230fb28f58490e57d4084462fde25b20c0e7afc6baf0fac6d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-41ad6244 | sha256-b557726a6d86a2b39fa7da0c7ebc52b0e09cffd8808957e0d5ddc71b5efd2c48 |
