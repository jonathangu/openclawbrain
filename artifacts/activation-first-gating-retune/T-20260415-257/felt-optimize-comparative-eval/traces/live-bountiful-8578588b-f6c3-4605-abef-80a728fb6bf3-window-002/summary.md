# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d264077d440eb71a4733b77855076b6c8ed4b150a58a0fe30d7cf9c384f3d83`
- fixture hash: `sha256-f8dd49edf14b538ae37136d1260c31d4ab4f9bbd2ad10ec02ad026ec49c5e356`
- score hash: `sha256-1e4888aaf49b1b6ceae392e1c2203f58996b61527ecfb4b54bafeef396e7da73`
- bundle hash: `sha256-0eda716a52a32ef1ee79d05c36d8a8f6b6dfcb22baf89902e93d10167e8d99f7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0ec5243a79ec6792abfa1dadf6e65f10d6e160b483bb5f843046d875ea86d177 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-94642c98cee7b9a89e1ab78d785275005bfa008bb0492778aaf065dd1b83e354 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-c747679d98a0a94b04758c5fdaf508e5ceb3a0dc7927b3cc1f38670db6922372 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-906429413fa0b25cfbbd9f3d8e67bdf6b60670cfe4d6db45783c3716ee84f505 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-c683a6c6 | sha256-63a04ed1f2f4b86f45f1a841b4cfb461b219bed0ad8935016c59f255c3ddebdd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-c683a6c6 | sha256-31b3aeecc1213e2403deba57f4890a170bc22ba7acc414f566fad41f4dc44a13 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c683a6c6 | sha256-7b95b0ef89a2b54985e52cc5278a72a2b78e1c1b488dfba216eeef9873fb41f9 |
