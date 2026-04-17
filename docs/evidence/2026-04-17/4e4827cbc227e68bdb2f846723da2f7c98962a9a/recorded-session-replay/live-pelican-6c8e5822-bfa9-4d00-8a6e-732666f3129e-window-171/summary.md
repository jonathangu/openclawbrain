# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f2aad77541ac9575f5e5ca17b331150d26a5ffdab9f43024542cda1cc603e5be`
- fixture hash: `sha256-bd1f8b0e0683d35bf0b6cddabbcb17bfbeff749dd6d56a3da4fa75988fc68560`
- score hash: `sha256-853b5f8779868053917e757d064c1bbd9c86f7d5dcc917acfe6143c50b067704`
- bundle hash: `sha256-84fbbf9a96818b6e62cd3bba8020b731aed1cbaffcd91ffa50910b0bd62e9194`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b6dcd51a56bbf9edfb3ea54756a6521b5761e2fe2a8b04b095719a90cd986e9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6db1443cadd7c7975919e19b51eae9a72752f76d1fa7e282ae78497d043d1c3f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dbb82dba49047fadcab2da536df2b65a0c88fd218d7598b06c5695ff81684350 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1cf1631ca6721804007d68519516136da9ab3f9097495dab12b51dc626f04133 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3198759a | sha256-e580a381f7e08a1da1b251213ba5ae6a8f6b1e2fd5696bc34ace7951a1e602dc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3198759a | sha256-da52ce648ad5417046c951fc618ffb8be78026e5f22d3169443c0a698632321e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d8e1b577 | sha256-8c892c3ab3ded9d0b6069d7562774eb9c66494617c36a71e026a885c5ff8039a |
